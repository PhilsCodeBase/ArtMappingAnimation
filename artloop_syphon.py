#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArtLoop -> Syphon Bruecke (macOS)
=================================
Empfaengt das Live-Bild von ArtLoop (Browser) und stellt es als Syphon-Quelle
"ArtLoop" bereit, z. B. fuer TouchDesigner (Syphon Spout In TOP), Resolume, MadMapper, OBS.

Einmalig installieren (Terminal):
    python3 -m pip install "syphon-python[numpy]"

Starten:
    python3 artloop_syphon.py

Danach im Browser http://127.0.0.1:8765 oeffnen (ArtLoop kommt direkt vom Hilfsprogramm)
und dort auf "Syphon" > "Senden starten" klicken.
In TouchDesigner: Syphon Spout In TOP anlegen, Sender Name = "ArtLoop".

Optionen:
    --name NAME   Name der Syphon-Quelle (Standard: ArtLoop)
    --port PORT   Port (Standard: 8765, muss mit ArtLoop uebereinstimmen)
    --flip        Bild vertikal spiegeln (falls es auf dem Kopf steht)
    --bgr         Rot/Blau tauschen (falls Farben vertauscht sind)
    --test        Ohne Syphon testen: speichert das erste Bild als artloop_test.ppm
"""
import argparse
import asyncio
import os
import base64
import hashlib
import struct
import sys
import time

if sys.version_info < (3, 10):
    print("Dieses Python ist zu alt (%d.%d). syphon-python braucht Python 3.10 oder neuer." % sys.version_info[:2])
    print("Loesung: aktuelles Python von https://www.python.org/downloads/ installieren,")
    print("dann im Terminal:  python3.12 -m pip install \"syphon-python[numpy]\"")
    print("und starten mit:   python3.12 artloop_syphon.py")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("numpy fehlt. Bitte installieren:  python3 -m pip install \"syphon-python[numpy]\"")
    sys.exit(1)

GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
HERE = os.path.dirname(os.path.abspath(__file__))
APP_B64 = ""  # wird beim Download aus ArtLoop automatisch mit der App gefuellt
STATIC = {"/manifest.webmanifest": "application/manifest+json", "/sw.js": "text/javascript",
          "/icon-192.png": "image/png", "/icon-512.png": "image/png", "/apple-touch-icon.png": "image/png"}
PAGES_ORIGIN = "https://philscodebase.github.io"
MAX_SIDE = 8192  # groesste erlaubte Bildkante in Pixeln
MAX_MESSAGE = 16 + MAX_SIDE * MAX_SIDE * 4


class FrameTooLarge(Exception):
    pass


def allowed_origins(port):
    """Nur ArtLoop selbst darf senden: vom Hilfsprogramm ausgeliefert oder von der GitHub-Seite."""
    return {"http://127.0.0.1:%d" % port, "http://localhost:%d" % port, "http://[::1]:%d" % port, PAGES_ORIGIN}


def app_html():
    """ArtLoop-Seite: eingebettet oder index.html / ArtLoop.html neben diesem Skript."""
    if APP_B64:
        return base64.b64decode(APP_B64)
    for name in ("index.html", "ArtLoop.html"):
        f = os.path.join(HERE, name)
        if os.path.isfile(f):
            with open(f, "rb") as fh:
                return fh.read()
    return None


def http_response(writer, code, ctype, body):
    head = ("HTTP/1.1 %s\r\nContent-Type: %s\r\nContent-Length: %d\r\n"
            "Cache-Control: no-store\r\nConnection: close\r\n\r\n" % (code, ctype, len(body)))
    writer.write(head.encode() + body)


class SyphonOut:
    def __init__(self, name):
        try:
            import syphon
            from syphon.utils.numpy import copy_image_to_mtl_texture
            from syphon.utils.raw import create_mtl_texture
        except ImportError as e:
            print("syphon-python fehlt oder laesst sich nicht laden (%s)." % e)
            print("Installieren mit:   %s -m pip install \"syphon-python[numpy]\"" % (os.path.basename(sys.executable) or "python3"))
            print("(benoetigt macOS 12+ und Python 3.10+)")
            sys.exit(1)
        self._copy = copy_image_to_mtl_texture
        self._create = create_mtl_texture
        self.server = syphon.SyphonMetalServer(name)
        self.tex = None
        self.size = None

    def publish(self, img):
        h, w = img.shape[:2]
        if self.size != (w, h):
            self.tex = self._create(self.server.device, w, h)
            self.size = (w, h)
        self._copy(img, self.tex)
        self.server.publish_frame_texture(self.tex)

    def stop(self):
        try:
            self.server.stop()
        except Exception:
            pass


class TestOut:
    """Ohne Syphon: zaehlt Bilder und speichert das erste als PPM."""
    def __init__(self, name):
        self.saved = False

    def publish(self, img):
        if not self.saved:
            h, w = img.shape[:2]
            with open("artloop_test.ppm", "wb") as f:
                f.write(b"P6 %d %d 255\n" % (w, h))
                f.write(np.ascontiguousarray(img[:, :, :3]).tobytes())
            self.saved = True
            print("Testbild gespeichert: artloop_test.ppm (%dx%d)" % (w, h))

    def stop(self):
        pass


def unmask(data, mask):
    n = len(data)
    pad = (-n) % 4
    if pad:
        data = data + b"\0" * pad
    arr = np.frombuffer(data, dtype=np.uint32)
    m = np.frombuffer(mask, dtype=np.uint32)[0]
    return (arr ^ m).view(np.uint8)[:n]


async def read_frame(reader):
    b = await reader.readexactly(2)
    fin = bool(b[0] & 0x80)
    op = b[0] & 0x0F
    masked = bool(b[1] & 0x80)
    ln = b[1] & 0x7F
    if ln == 126:
        ln = struct.unpack(">H", await reader.readexactly(2))[0]
    elif ln == 127:
        ln = struct.unpack(">Q", await reader.readexactly(8))[0]
    if ln > MAX_MESSAGE:
        raise FrameTooLarge()
    mask = await reader.readexactly(4) if masked else None
    data = await reader.readexactly(ln) if ln else b""
    payload = unmask(data, mask) if mask else np.frombuffer(data, dtype=np.uint8)
    return fin, op, payload


def ws_send(writer, op, payload=b""):
    n = len(payload)
    if n < 126:
        hdr = struct.pack(">BB", 0x80 | op, n)
    elif n < 65536:
        hdr = struct.pack(">BBH", 0x80 | op, 126, n)
    else:
        hdr = struct.pack(">BBQ", 0x80 | op, 127, n)
    writer.write(hdr + payload)


def make_handler(out, args):
    async def handle(reader, writer):
        peer = writer.get_extra_info("peername")
        try:
            req = await reader.readuntil(b"\r\n\r\n")
        except Exception:
            writer.close()
            return
        lines = req.decode("latin-1").split("\r\n")
        try:
            path = lines[0].split(" ")[1].split("?")[0]
        except IndexError:
            path = "/"
        headers = {}
        for line in lines[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        key = headers.get("sec-websocket-key")
        if not key:
            if path in ("/", "/index.html"):
                html = app_html()
                if html:
                    http_response(writer, "200 OK", "text/html; charset=utf-8", html)
                else:
                    http_response(writer, "200 OK", "text/html; charset=utf-8",
                                  "<h2>ArtLoop Syphon-Br&uuml;cke l&auml;uft.</h2><p>Lege ArtLoop.html bzw. index.html in denselben Ordner wie artloop_syphon.py, "
                                  "dann erscheint ArtLoop hier.</p>".encode())
            elif path in STATIC and os.path.isfile(os.path.join(HERE, path[1:])):
                with open(os.path.join(HERE, path[1:]), "rb") as fh:
                    http_response(writer, "200 OK", STATIC[path], fh.read())
            elif path == "/status":
                http_response(writer, "200 OK", "text/plain", b"ArtLoop Syphon-Bruecke laeuft.\n")
            else:
                http_response(writer, "404 Not Found", "text/plain", b"nicht gefunden\n")
            await writer.drain()
            writer.close()
            return
        origin = headers.get("origin", "")
        if origin.lower() not in allowed_origins(args.port):
            print("Verbindung abgelehnt, fremde Herkunft: %r" % origin)
            http_response(writer, "403 Forbidden", "text/plain", b"Herkunft nicht erlaubt\n")
            await writer.drain()
            writer.close()
            return
        accept = base64.b64encode(hashlib.sha1(key.encode() + GUID).digest()).decode()
        writer.write(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                      "Connection: Upgrade\r\nSec-WebSocket-Accept: %s\r\n\r\n" % accept).encode())
        await writer.drain()
        print("ArtLoop verbunden", peer)
        parts, size = [], 0
        frames, t0 = 0, time.time()
        try:
            while True:
                fin, op, data = await read_frame(reader)
                if op == 8:
                    ws_send(writer, 8)
                    await writer.drain()
                    break
                if op == 9:
                    ws_send(writer, 10, data.tobytes())
                    await writer.drain()
                    continue
                if op not in (0, 1, 2):
                    continue
                parts.append(data)
                size += len(data)
                if size > MAX_MESSAGE:
                    raise FrameTooLarge()
                if not fin:
                    continue
                msg = parts[0] if len(parts) == 1 else np.concatenate(parts)
                parts, size = [], 0
                if op == 1 or len(msg) < 16 or msg[:4].tobytes() != b"ALP1":
                    continue
                w, h, flags = struct.unpack("<III", msg[4:16].tobytes())
                if not (0 < w <= MAX_SIDE and 0 < h <= MAX_SIDE) or w * h * 4 + 16 > len(msg):
                    continue
                img = msg[16:16 + w * h * 4].reshape(h, w, 4)
                if (flags & 1) != (1 if args.flip else 0):
                    img = img[::-1]
                if args.bgr:
                    img = img[:, :, [2, 1, 0, 3]]
                out.publish(np.ascontiguousarray(img))
                frames += 1
                dt = time.time() - t0
                if dt >= 3:
                    print("  %dx%d  %.1f fps" % (w, h, frames / dt))
                    frames, t0 = 0, time.time()
        except FrameTooLarge:
            print("Verbindung getrennt: Datenpaket zu gross")
            ws_send(writer, 8, struct.pack(">H", 1009))
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            print("ArtLoop getrennt")
            writer.close()
    return handle


async def main():
    ap = argparse.ArgumentParser(description="ArtLoop -> Syphon Bruecke")
    ap.add_argument("--name", default="ArtLoop")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--bgr", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    out = TestOut(args.name) if args.test else SyphonOut(args.name)
    handler = make_handler(out, args)
    try:
        server = await asyncio.start_server(handler, ["127.0.0.1", "::1"], args.port)
    except OSError:
        try:
            server = await asyncio.start_server(handler, "127.0.0.1", args.port)
        except OSError as e:
            print("Port %d ist belegt oder gesperrt (%s). Laeuft das Hilfsprogramm schon in einem anderen Fenster?" % (args.port, e))
            print("Sonst anderen Port waehlen:  python3 artloop_syphon.py --port 8766  (und in ArtLoop denselben Port eintragen)")
            return
    print("Syphon-Quelle \"%s\" bereit." % args.name)
    print("")
    print("  >>> Jetzt im Browser oeffnen:  http://127.0.0.1:%d" % args.port)
    print("      dann in ArtLoop: Syphon > Senden starten")
    if not app_html():
        print("      (ArtLoop.html bzw. index.html in diesen Ordner legen, damit die Seite dort erscheint)")
    print("")
    print("Beenden: Ctrl+C")
    try:
        async with server:
            await server.serve_forever()
    finally:
        out.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBeendet.")
