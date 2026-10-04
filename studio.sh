#!/usr/bin/env sh
# Cabeceando Studio: the desktop app for list videos (Fedora, CachyOS, any Linux).
#   ./studio.sh                 start it and open the browser
#   MPT_STUDIO_PORT=8610 ./studio.sh
#   ./studio.sh --install       add "Cabeceando Studio" to the applications menu

CURRENT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$CURRENT_DIR${PYTHONPATH:+:$PYTHONPATH}"
HOST="${MPT_STUDIO_HOST:-127.0.0.1}"
PORT="${MPT_STUDIO_PORT:-8600}"

if [ "$1" = "--install" ]; then
  APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
  mkdir -p "$APPS"
  cat > "$APPS/cabeceando-studio.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Cabeceando Studio
Comment=Videos educativos automáticos explicados para nutrias
Exec=sh "$CURRENT_DIR/studio.sh"
Icon=$CURRENT_DIR/resource/characters/nutria/personaje/saludando.png
Terminal=false
Categories=AudioVideo;Video;Education;
DESKTOP
  chmod +x "$APPS/cabeceando-studio.desktop"
  command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS" >/dev/null 2>&1
  echo "Cabeceando Studio added to the applications menu ($APPS/cabeceando-studio.desktop)"
  exit 0
fi

if [ -x "$CURRENT_DIR/.venv/bin/python" ]; then
  PYTHON="$CURRENT_DIR/.venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
  (cd "$CURRENT_DIR" && uv sync --frozen) || exit 1
  PYTHON="$CURRENT_DIR/.venv/bin/python"
else
  echo "Install the dependencies first: uv sync --frozen (https://docs.astral.sh/uv/)"
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Tip: install ffmpeg (Fedora: sudo dnf install ffmpeg; CachyOS: sudo pacman -S ffmpeg)"
fi

# The first free port from $PORT.
PORT=$("$PYTHON" - "$HOST" "$PORT" <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
for candidate in range(port, port + 50):
    with socket.socket() as sock:
        try:
            sock.bind((host, candidate))
        except OSError:
            continue
        print(candidate)
        break
PY
)

URL="http://$HOST:$PORT"
echo "Cabeceando Studio: $URL"
( sleep 3; command -v xdg-open >/dev/null 2>&1 && xdg-open "$URL" >/dev/null 2>&1 ) &
cd "$CURRENT_DIR" && exec "$PYTHON" -m streamlit run "$CURRENT_DIR/studio/Studio.py" \
  --server.address="$HOST" \
  --server.port="$PORT" \
  --server.headless=true \
  --browser.gatherUsageStats=false \
  --client.toolbarMode=minimal \
  --theme.base=light \
  --theme.primaryColor="#FF4F5E" \
  --theme.backgroundColor="#FFFDFB" \
  --theme.secondaryBackgroundColor="#FFF0F1" \
  --theme.textColor="#1B1B2F"
