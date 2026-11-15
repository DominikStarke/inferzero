#!/usr/bin/env bash
# inferzero on Linux: checks the system, lists the catalog's models (downloaded or not), downloads one into the Hugging
# Face cache with curl, starts bin/serverz and opens the web UI. Bash and the system's tools only (curl, sha256sum):
# no Python, as the rest of inferzero. The Windows launcher is zero-launcher.cmd.
#   ./zero-launcher.sh                    the menu
#   ./zero-launcher.sh --check            what the system has and lacks (GPU runtime, device access, CPU, RAM)
#   ./zero-launcher.sh --list             the catalog, with what is downloaded
#   ./zero-launcher.sh --download 4b      download a catalog model (no server)
#   ./zero-launcher.sh --model 4b         download if needed, then start the server (also a path to a .gguf)
#   ./zero-launcher.sh --model 4b --install-service   a systemd user service for it instead of starting it here
# Options: --port N (8090), --host H (127.0.0.1), --gpus N, --mtp (with --model FILE), --no-browser, --yes (no
# questions), --service-name NAME (inferzero), --replace-service; anything after `--` goes to serverz as it is
# (e.g. -- --slots 2; not --port / --host, which the launcher must know).
# The catalog below follows scripts/model-catalog.ps1 (tests/check_catalog_sync.sh checks that they agree). Flash
# Next's AutoRound import and the speech models are setup.ps1's on Windows and not offered here.
# (A symlink to this file, e.g. in ~/bin, is followed to the package; a relative --model path stays relative to where
#  you ran it.)

SELF=$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || printf '%s' "${BASH_SOURCE[0]}")
ROOT=$(cd "$(dirname "$SELF")" && pwd)
SERVERZ="$ROOT/bin/serverz"

# ---- the catalog: key|name|repo|GB|disk GB|RAM GB|mtp|files (space-separated)|drafter repo|drafter file|extra repo|extra file|extra into
CATALOG=(
"4b|Qwen3.5 4B (fastest; MTP, reads images)|unsloth/Qwen3.5-4B-MTP-GGUF|3.7||16|1|Qwen3.5-4B-UD-Q4_K_XL.gguf mmproj-F16.gguf|||||"
"9b|Qwen3.5 9B (MTP, reads images)|unsloth/Qwen3.5-9B-MTP-GGUF|7.1||16|1|Qwen3.5-9B-UD-Q4_K_XL.gguf mmproj-F16.gguf|||||"
"35b|Qwen3.6 35B-A3B (fast mixture of experts; MTP, reads images)|unsloth/Qwen3.6-35B-A3B-MTP-GGUF|19.1||32|1|Qwen3.6-35B-A3B-UD-IQ4_XS.gguf mmproj-F16.gguf|||||"
"27b|Qwen3.8 27B (best answers, slowest; MTP, reads images)|ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF|13.1||32|1|Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf mmproj-Qwen3.8-27B-BF16.gguf|||||"
"flash|Qwen3.8 Flash Next (512 experts on the SSD; reads images)|ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF|84.5|141|32|0|IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf mmproj-Qwen3.8-Flash-Next-f16.gguf|||bartowski/Qwen3.8-Flash-Next-GGUF|mmproj-Qwen3.8-Flash-Next-f16.gguf|IQ3_S"
"g12|Gemma 4 12B (reads images; drafter)|google/gemma-4-12B-it-qat-q4_0-gguf|7.4||32|1|gemma-4-12b-it-qat-q4_0.gguf mmproj-gemma-4-12b-it-qat-q4_0.gguf mtp-gemma-4-12B-it.gguf|unsloth/gemma-4-12B-it-qat-GGUF|mtp-gemma-4-12B-it.gguf|||"
"g31|Gemma 4 31B (Google's largest dense Gemma; slow; drafter)|google/gemma-4-31B-it-qat-q4_0-gguf|19.1||32|1|gemma-4-31B_q4_0-it.gguf gemma-4-31B-it-mmproj.gguf mtp-gemma-4-31B-it.gguf|unsloth/gemma-4-31B-it-qat-GGUF|mtp-gemma-4-31B-it.gguf|||"
"e4b|Gemma 4 E4B (images, audio; drafter)|google/gemma-4-E4B-it-qat-q4_0-gguf|6.2||16|1|gemma-4-E4B_q4_0-it.gguf gemma-4-E4B-it-mmproj.gguf mtp-gemma-4-E4B-it.gguf|unsloth/gemma-4-E4B-it-qat-GGUF|mtp-gemma-4-E4B-it.gguf|||"
)
# the cache variables the server and the downloader read (a systemd unit gets them too); never HF_TOKEN
CACHE_ENV=(HF_HUB_CACHE HF_HOME XDG_CACHE_HOME HF_TOKEN_PATH)

say() { printf '%s\n' "$*"; }
err() { printf '%s\n' "$*" >&2; }

# m_field KEY N: field N (0-based) of a catalog entry
m_field() {
  local e
  for e in "${CATALOG[@]}"; do
    if [ "${e%%|*}" = "$1" ]; then IFS='|' read -r -a _f <<< "$e"; printf '%s' "${_f[$2]}"; return 0; fi
  done
  return 1
}
m_known() { m_field "$1" 0 >/dev/null; }

hub_dir() {  # the Hugging Face cache, as the engine (src/platform.h hf_home_hub) finds it
  if [ -n "${HF_HUB_CACHE:-}" ]; then printf '%s' "$HF_HUB_CACHE"
  elif [ -n "${HF_HOME:-}" ]; then printf '%s/hub' "$HF_HOME"
  elif [ "${XDG_CACHE_HOME:0:1}" = "/" ]; then printf '%s/huggingface/hub' "$XDG_CACHE_HOME"
  else printf '%s/.cache/huggingface/hub' "$HOME"; fi
}
repo_root() { printf '%s/models--%s' "$(hub_dir)" "${1//\//--}"; }

snapshots() {  # the repository's snapshot folders: the one refs/main points at first, then the others
  local root main d
  root=$(repo_root "$1")
  if [ -f "$root/refs/main" ]; then
    main="$root/snapshots/$(tr -d '[:space:]' < "$root/refs/main")"
    [ -d "$main" ] && printf '%s\n' "$main"
  fi
  [ -d "$root/snapshots" ] || return 0
  for d in "$root"/snapshots/*/; do
    d=${d%/}
    [ -d "$d" ] && [ "$d" != "${main:-}" ] && printf '%s\n' "$d"
  done
}

has_file() {  # snapshot, path: there, or (a file copied in from another repository) by its name anywhere inside
  [ -f "$1/$2" ] && return 0
  [ -n "$(find "$1" -name "$(basename "$2")" -type f -print -quit 2>/dev/null)" ]
}

model_path() {  # the first file of a downloaded catalog model (a snapshot that holds all its files), or nothing
  local key=$1 repo files d f ok
  repo=$(m_field "$key" 2); files=$(m_field "$key" 7)
  while IFS= read -r d; do
    [ -n "$d" ] || continue
    ok=1
    for f in $files; do has_file "$d" "$f" || { ok=0; break; }; done
    if [ $ok -eq 1 ]; then printf '%s/%s' "$d" "${files%% *}"; return 0; fi
  done < <(snapshots "$repo")
  return 1
}

ram_gb() { awk '/^MemTotal:/ { printf "%d", $2 / 1048576 }' /proc/meminfo 2>/dev/null || printf '0'; }

have_lib() {  # a shared library the dynamic loader finds
  ldconfig -p 2>/dev/null | grep -q "$1" && return 0
  local d
  for d in /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu /usr/local/lib ${LD_LIBRARY_PATH//:/ }; do
    [ -e "$d/$1" ] && return 0
  done
  return 1
}

# check: one line per need, "ok    ..." or "MISS  ..."; returns 1 when something is missing
CHECK_LINES=()
run_check() {
  CHECK_LINES=()
  local bad=0 node nodes=() n
  _c() { if [ "$1" -eq 0 ]; then CHECK_LINES+=("  ok    $2"); else CHECK_LINES+=("  MISS  $2"); bad=1; fi; }
  [ "$(uname -s)" = Linux ]; _c $? "Linux (this launcher is for Linux; on Windows use zero-launcher.cmd)"
  [ -x "$SERVERZ" ]; _c $? "bin/serverz built ($SERVERZ; ./build.sh builds it)"
  grep -m1 '^flags' /proc/cpuinfo 2>/dev/null | grep -qw avx2 && grep -m1 '^flags' /proc/cpuinfo | grep -qw fma &&
    grep -m1 '^flags' /proc/cpuinfo | grep -qw f16c
  _c $? "CPU with AVX2, FMA and F16C"
  have_lib libze_loader.so.1; _c $? "libze_loader.so.1 (sudo apt install libze1)"
  have_lib libOpenCL.so.1; _c $? "libOpenCL.so.1 (sudo apt install ocl-icd-libopencl1)"
  have_lib libgomp.so.1; _c $? "libgomp.so.1 (sudo apt install libgomp1)"
  ls /etc/OpenCL/vendors/ 2>/dev/null | grep -qi intel
  _c $? "Intel's OpenCL driver (/etc/OpenCL/vendors/intel*.icd; sudo apt install intel-opencl-icd libze-intel-gpu1, or Intel's packages for Battlemage cards: docs/linux-port.md)"
  for n in /dev/dri/renderD*; do [ -e "$n" ] && nodes+=("$n"); done
  [ ${#nodes[@]} -eq 0 ] && [ -e /dev/dxg ] && nodes=(/dev/dxg)  # (WSL2: the Windows driver's GPU, no render node)
  [ ${#nodes[@]} -gt 0 ]; _c $? "a GPU render node (/dev/dri/renderD*, or /dev/dxg in WSL2)"
  if [ ${#nodes[@]} -gt 0 ]; then
    local acc=1
    for node in "${nodes[@]}"; do [ -r "$node" ] && [ -w "$node" ] && acc=0; done
    _c $acc "access to the render node (sudo usermod -aG render \$USER, then log in again)"
  fi
  local r; r=$(ram_gb)
  [ "${r:-0}" -ge 15 ]; _c $? "RAM ${r} GB (16 GB for the 4B / 9B / E4B, 32 GB for the others)"
  return $bad
}
print_check() { local rc=0; run_check || rc=1; printf '%s\n' "${CHECK_LINES[@]}"; return $rc; }

# ---- downloads, with curl into the Hugging Face cache layout (snapshots/<commit>/<file>, refs/main); each file's
# size and, for large (LFS) files, its SHA-256 from the repository's own headers, checked before it takes its name
hf_token() {
  if [ -n "${HF_TOKEN:-}" ]; then printf '%s' "$HF_TOKEN"; return; fi
  local f="${HF_HOME:-$HOME/.cache/huggingface}/token"
  [ -f "$f" ] && tr -d '[:space:]' < "$f"
}
# hf_head REPO REV FILE: the headers of the file's first answer (not followed), lower-cased names
hf_head() {
  local tok a=(-sS -I --max-time 60 --noproxy '' -A 'inferzero-launcher')
  tok=$(hf_token)
  [ -n "$tok" ] && a+=(-H "Authorization: Bearer $tok")
  curl "${a[@]}" "https://huggingface.co/$1/resolve/$2/$3" 2>/dev/null | tr -d '\r' | sed 's/^\([^:]*\):/\L\1:/'
}
hdr() { sed -n "s/^$1: *//p" | head -1 | tr -d '"'; }

safe_rel() {  # a repository path that stays inside the snapshot (no .., no absolute path)
  case "/$1/" in */../*|*//*) return 1;; esac
  [ "${1:0:1}" != "/" ]
}

# hf_get REPO FILE...: downloads into the cache; prints the snapshot folder on the last line
hf_get() {
  local repo=$1; shift
  local h commit f size sha snap dst part have tok need=0 free
  h=$(hf_head "$repo" main "$1")
  commit=$(printf '%s\n' "$h" | hdr x-repo-commit)
  if [ -z "$commit" ]; then
    case "$(printf '%s\n' "$h" | head -1)" in
      *" 401"*|*" 403"*) err "$repo needs you to sign in and accept its terms on huggingface.co first, then a token: set HF_TOKEN (a read token from https://huggingface.co/settings/tokens) and run this again";;
      *) err "cannot reach huggingface.co for $repo (or it has no file $1): check the internet connection, then run this again";;
    esac
    return 1
  fi
  snap="$(repo_root "$repo")/snapshots/$commit"
  mkdir -p "$snap" "$(repo_root "$repo")/refs" || return 1
  printf '%s' "$commit" > "$(repo_root "$repo")/refs/main"
  local -A SIZE SHA
  for f in "$@"; do
    safe_rel "$f" || { err "$f is not a path inside the repository"; return 1; }
    h=$(hf_head "$repo" "$commit" "$f")
    size=$(printf '%s\n' "$h" | hdr x-linked-size); [ -n "$size" ] || size=$(printf '%s\n' "$h" | hdr content-length)
    sha=$(printf '%s\n' "$h" | hdr x-linked-etag)
    [ -n "$size" ] || { err "$repo has no file $f"; return 1; }
    SIZE[$f]=$size; SHA[$f]=$sha
    dst="$snap/$f"; have=0
    if [ -f "$dst" ]; then have=$(stat -c %s "$dst"); elif [ -f "$dst.part" ]; then have=$(stat -c %s "$dst.part"); fi
    [ "$have" -lt "$size" ] && need=$((need + size - have))
  done
  free=$(df -PB1 "$snap" 2>/dev/null | awk 'NR == 2 { print $4 }')
  if [ -n "$free" ] && [ "$need" -gt "$free" ]; then
    err "$(awk -v n="$need" -v f="$free" -v p="$snap" 'BEGIN { printf "these files need %.1f GB more, and %s has %.1f GB free: make room, or set HF_HOME to a folder on a larger drive", n / 1e9, p, f / 1e9 }')"
    return 1
  fi
  [ "$need" -gt 0 ] && awk -v n="$need" -v f="${free:-0}" 'BEGIN { printf "    %.1f GB to download (%.1f GB free)\n", n / 1e9, f / 1e9 }' >&2
  tok=$(hf_token)
  for f in "$@"; do
    dst="$snap/$f"; part="$dst.part"; size=${SIZE[$f]}
    mkdir -p "$(dirname "$dst")"
    if [ -f "$dst" ] && [ "$(stat -c %s "$dst")" = "$size" ]; then err "    $f is there"; continue; fi
    have=0; [ -f "$part" ] && have=$(stat -c %s "$part")
    if [ "$have" -gt "$size" ]; then rm -f "$part"; have=0; fi
    if [ "$have" -lt "$size" ]; then
      awk -v f="$f" -v s="$size" -v h="$have" 'BEGIN { printf "    %s (%.2f GB%s)\n", f, s / 1e9, h ? sprintf(", resuming at %.2f GB", h / 1e9) : "" }' >&2
      # (--speed-time: a connection that stalls ends with an error instead of waiting forever; run again to resume)
      local a=(-L --fail --retry 5 --retry-delay 3 --speed-limit 1 --speed-time 120 --progress-bar -A 'inferzero-launcher' -o "$part")
      [ "$have" -gt 0 ] && a+=(-C -)
      [[ "${ARC_HF_RATE:-}" =~ ^[0-9]+[KkMmGg]?$ ]] && a+=(--limit-rate "$ARC_HF_RATE")
      [ -n "$tok" ] && a+=(-H "Authorization: Bearer $tok")
      if ! curl "${a[@]}" "https://huggingface.co/$repo/resolve/$commit/$f"; then
        err "downloading $f failed: run this again, it goes on where it stopped"; return 1
      fi
    fi
    have=$(stat -c %s "$part")
    [ "$have" = "$size" ] || { err "$f is $have bytes, the repository says $size: run this again"; return 1; }
    if [ ${#SHA[$f]} -eq 64 ]; then
      err "    checking $f (SHA-256)..."
      if [ "$(sha256sum "$part" | cut -d' ' -f1)" != "${SHA[$f],,}" ]; then
        rm -f "$part"; err "$f arrived damaged (SHA-256 differs): run this again to download it anew"; return 1
      fi
    fi
    mv -f "$part" "$dst"
  done
  printf '%s\n' "$snap"
}

copy_into() {  # atomic: a copy stopped halfway never looks like the finished file
  cp -f "$1" "$2.incomplete" && mv -f "$2.incomplete" "$2"
}

download() {  # KEY YES
  local key=$1 yes=$2 name repo files gb own f snap dr df xr xf xi xs src
  name=$(m_field "$key" 1); repo=$(m_field "$key" 2); gb=$(m_field "$key" 3); files=$(m_field "$key" 7)
  dr=$(m_field "$key" 8); df=$(m_field "$key" 9); xr=$(m_field "$key" 10); xf=$(m_field "$key" 11); xi=$(m_field "$key" 12)
  command -v curl >/dev/null || { err "curl not found (sudo apt install curl)"; return 1; }
  if [ "$yes" != 1 ]; then
    local ans; read -r -p "Download $name (~$gb GB) into $(hub_dir)? [y/N] " ans || return 1
    [ "${ans,,}" = y ] || return 1
  fi
  own=()
  for f in $files; do [ "$f" = "$df" ] || [ "$f" = "$xf" ] || own+=("$f"); done
  snap=$(hf_get "$repo" "${own[@]}" | tail -1) || return 1
  [ -d "$snap" ] || return 1
  # (a file from another repository copied next to the model, where the engine looks for it: Gemma's drafter, Flash
  #  Next's image projector)
  if [ -n "$dr" ]; then
    xs=$(hf_get "$dr" "$df" | tail -1) || return 1
    copy_into "$xs/$df" "$snap/$(basename "$df")" || return 1
  fi
  if [ -n "$xr" ]; then
    xs=$(hf_get "$xr" "$xf" | tail -1) || return 1
    mkdir -p "$snap/$xi"; copy_into "$xs/$xf" "$snap/$xi/$(basename "$xf")" || return 1
  fi
  if ! model_path "$key" >/dev/null; then
    err "$name: some of its files are still missing (a new commit of the repository during the download?); run it again"
    return 1
  fi
}

# ---- the server
exposed() { case "$1" in ''|127.0.0.1|localhost|::1) return 1;; esac; return 0; }
port_busy() {  # HOST PORT: something already answers there (another server would answer our /health)
  local h=${1:-127.0.0.1}
  [ "$h" = 0.0.0.0 ] && h=127.0.0.1
  [ "$h" = "::" ] && h=::1
  (exec 3<>"/dev/tcp/$h/$2") 2>/dev/null
}

server_cmd() {  # PATH MTP: fills CMD
  CMD=("$SERVERZ" -m "$1" --port "$PORT" --ctx auto --think-budget 8192)
  [ "$2" = 1 ] && CMD+=(--mtp)
  [ -n "$HOST" ] && CMD+=(--host "$HOST")
  [ "${GPUS:-0}" -gt 1 ] && CMD+=(--gpus "$GPUS")
  CMD+=("${EXTRA[@]}")
}

run_server() {
  local host=${HOST:-127.0.0.1} shown url pid opened=0 rc
  case "$host" in 0.0.0.0) shown=127.0.0.1;; ::) shown=::1;; *) shown=$host;; esac
  case "$shown" in *:*) url="http://[$shown]:$PORT/";; *) url="http://$shown:$PORT/";; esac
  if port_busy "$host" "$PORT"; then say "port $PORT is in use (another server?): stop it, or choose another with --port"; return 1; fi
  exposed "$HOST" && say "note: --host $HOST makes the server reachable from your network; it has no password of its own"
  say "starting: ${CMD[*]}"
  (cd "$ROOT" && exec "${CMD[@]}") &
  pid=$!
  # (passed on: Ctrl+C reaches the server from the terminal too, which is harmless; a SIGINT or SIGTERM sent to the
  #  launcher alone (kill, an IDE's stop, systemd) must reach it as well)
  trap 'kill -INT $pid 2>/dev/null' INT
  trap 'kill -TERM $pid 2>/dev/null' TERM
  while kill -0 "$pid" 2>/dev/null; do
    if [ $opened -eq 0 ] && curl -s --noproxy '*' --max-time 2 -o /dev/null -w '%{http_code}' "${url}health" 2>/dev/null | grep -q '^200$'; then
      opened=1
      say ""; say "ready: $url  (Ctrl+C stops the server)"
      if [ "$NO_BROWSER" != 1 ] && { [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; } && command -v xdg-open >/dev/null; then
        # (a session of its own: Ctrl+C or a closed terminal must not take the browser with it)
        setsid xdg-open "$url" >/dev/null 2>&1 < /dev/null &
      fi
    fi
    sleep 1
  done
  wait "$pid"; rc=$?
  trap - INT TERM
  return $rc
}

# ---- the systemd user service
service_unit() {  # prints the unit's path, or fails with a message when the name is not a plain unit name
  local name=${SERVICE_NAME%.service} base
  if ! [[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9_.@-]{0,127}$ ]]; then
    err "--service-name: use a simple unit name, without a path or whitespace"; return 1
  fi
  base=${XDG_CONFIG_HOME:-}
  [ "${base:0:1}" = "/" ] || base="$HOME/.config"
  printf '%s/systemd/user/%s.service' "$base" "$name"
}
service_refused() {  # a reason not to write the unit (checked before a download), or nothing
  local unit
  unit=$(service_unit) || return 0
  if { [ -e "$unit" ] || [ -L "$unit" ]; } && [ "$REPLACE_SERVICE" != 1 ]; then
    printf '%s already exists; left unchanged. Use --service-name NAME for a separate service, or --replace-service to replace this unit explicitly.' "$unit"
  fi
  return 0
}
# systemd's own quoting, not the shell's: each word in double quotes with \ and " escaped and its % specifiers doubled;
# in ExecStart= its $ variables too, which systemd expands there and not in Environment=
sq() { local c=$1; c=${c//\\/\\\\}; c=${c//\"/\\\"}; c=${c//%/%%}; [ "${2:-1}" = 1 ] && c=${c//\$/\$\$}; printf '"%s"' "$c"; }

install_service() {
  local unit why name d tmp existed=0 c k v exec_line="" env_lines="" nl=$'\n' cr=$'\r'
  why=$(service_refused)
  if [ -n "$why" ]; then say "$why"; return 1; fi
  unit=$(service_unit) || return 1
  name=$(basename "$unit" .service)
  for c in "${CMD[@]}"; do
    case "$c" in *"$nl"*|*"$cr"*) say "--install-service: a path, option or cache environment value holds a line break"; return 1;; esac
    exec_line+="$(sq "$c") "
  done
  for k in "${CACHE_ENV[@]}"; do
    v=${!k:-}
    [ -n "$v" ] || continue
    case "$v" in *"$nl"*|*"$cr"*) say "--install-service: a path, option or cache environment value holds a line break"; return 1;; esac
    env_lines+="Environment=$(sq "$k=$v" 0)$nl"
  done
  d=$(dirname "$unit")
  mkdir -p "$d" || return 1
  { [ -e "$unit" ] || [ -L "$unit" ]; } && existed=1
  tmp=$(mktemp "$d/.inferzero-XXXXXX") || return 1
  # (KillMode=control-group, the default: the server's children (terminal tool, MCP servers) end with it;
  #  WorkingDirectory: the package, where the server finds its kernels and web UI)
  printf '[Unit]\nDescription=inferzero server\nAfter=network.target\n\n[Service]\nWorkingDirectory=%s\n%sExecStart=%s\nRestart=on-failure\nRestartSec=5\n\n[Install]\nWantedBy=default.target\n' \
    "$(sq "$ROOT")" "$env_lines" "${exec_line% }" > "$tmp"
  if [ "$REPLACE_SERVICE" = 1 ]; then
    mv -fT "$tmp" "$unit" || { rm -f "$tmp"; return 1; }  # (replaces a symlink itself, never writes through it)
  elif ! ln "$tmp" "$unit" 2>/dev/null; then  # (an atomic create that never replaces a file)
    if [ -e "$unit" ] || [ -L "$unit" ]; then rm -f "$tmp"; say "$unit already exists; left unchanged."; return 1; fi
    # (a file system without hard links: an exclusive create instead, still never replacing a file)
    if ! (set -C; cat "$tmp" > "$unit") 2>/dev/null; then rm -f "$tmp"; say "$unit already exists; left unchanged."; return 1; fi
    rm -f "$tmp"
  else
    rm -f "$tmp"
  fi
  if [ $existed -eq 1 ]; then say "replaced $unit"; else say "wrote $unit"; fi
  exposed "$HOST" && say "note: --host $HOST makes the server reachable from your network; it has no password of its own"
  say "start it now and at every login:"
  say "  systemctl --user daemon-reload"
  say "  systemctl --user enable $name"
  say "  systemctl --user restart $name"
  say "its log: journalctl --user -u $name -f"
  say "(to keep it running while logged out: sudo loginctl enable-linger \$USER)"
}

start_key() {  # KEY: download if needed, then the server or the unit
  local key=$1 path why
  if [ "$INSTALL_SERVICE" = 1 ]; then  # (refused before any download, not after several GB of it)
    why=$(service_refused); [ -z "$why" ] || { say "$why"; return 1; }
    service_unit >/dev/null || return 1
  fi
  path=$(model_path "$key") || { download "$key" "$YES" || return 1; path=$(model_path "$key") || return 1; }
  server_cmd "$path" "$(m_field "$key" 6)"
  if [ "$INSTALL_SERVICE" = 1 ]; then install_service; else run_server; fi
}

list_models() {
  local e key p
  for e in "${CATALOG[@]}"; do
    IFS='|' read -r -a _f <<< "$e"; key=${_f[0]}
    if p=$(model_path "$key"); then
      printf '%-6s * %s, ~%s GB, %s GB RAM\n         %s\n' "$key" "${_f[1]}" "${_f[3]}" "${_f[5]}" "$p"
    else
      printf '%-6s   %s, ~%s GB, %s GB RAM\n' "$key" "${_f[1]}" "${_f[3]}" "${_f[5]}"
    fi
  done
}

menu() {
  local ram i=0 e mark fit s keys=()
  ram=$(ram_gb)
  say "inferzero: models (* downloaded)"; say ""
  for e in "${CATALOG[@]}"; do
    IFS='|' read -r -a _f <<< "$e"; i=$((i + 1)); keys+=("${_f[0]}")
    mark=" "; model_path "${_f[0]}" >/dev/null && mark="*"
    fit=""; [ "${ram:-0}" -gt 0 ] && [ $((ram + 1)) -lt "${_f[5]}" ] && fit="  (wants ${_f[5]} GB of RAM)"
    printf ' %2d %s %s, ~%s GB%s\n' "$i" "$mark" "${_f[1]}" "${_f[3]}" "$fit"
  done
  say ""; say "  c  check the system    q  quit"
  read -r -p $'\nmodel number: ' s || return 130
  s=${s,,}
  case "$s" in
    c) print_check; return $?;;
    q|'') return 0;;
  esac
  if ! [[ "$s" =~ ^[0-9]+$ ]] || [ "$s" -lt 1 ] || [ "$s" -gt ${#keys[@]} ]; then say "no model \"$s\""; return 2; fi
  start_key "${keys[$((s - 1))]}"
}

usage() { sed -n '2,17p' "$SELF" | sed 's/^# \{0,1\}//'; }

main() {
  local mode="" arg_model="" arg_dl=""
  PORT=8090; HOST=""; GPUS=0; NO_BROWSER=0; YES=0; MTP=0; INSTALL_SERVICE=0; SERVICE_NAME=inferzero
  REPLACE_SERVICE=0; EXTRA=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --check) mode=check;;
      --list) mode=list;;
      --download) arg_dl=${2:-}; shift; mode=download;;
      --model) arg_model=${2:-}; shift;;
      --mtp) MTP=1;;
      --port) PORT=${2:-}; shift;;
      --host) HOST=${2:-}; shift;;
      --gpus) GPUS=${2:-0}; shift;;
      --no-browser) NO_BROWSER=1;;
      --yes) YES=1;;
      --install-service) INSTALL_SERVICE=1;;
      --service-name) SERVICE_NAME=${2:-}; shift;;
      --replace-service) REPLACE_SERVICE=1;;
      -h|--help) usage; return 0;;
      --) shift; EXTRA=("$@"); break;;
      *) err "zero-launcher: unknown option $1 (--help lists them)"; return 2;;
    esac
    shift
  done
  local x
  for x in "${EXTRA[@]}"; do  # (the launcher polls the port it was given: those two go through its own options)
    case "${x%%=*}" in --port|--host) err "zero-launcher: ${x%%=*} after --: give it to the launcher itself (before --)"; return 2;; esac
  done
  [[ "$PORT" =~ ^[0-9]+$ ]] && [ "$PORT" -ge 1 ] && [ "$PORT" -le 65535 ] || { err "zero-launcher: --port: 1..65535"; return 2; }
  [[ "$GPUS" =~ ^[0-9]+$ ]] || { err "zero-launcher: --gpus: a number"; return 2; }
  [ "$REPLACE_SERVICE" = 1 ] && [ "$INSTALL_SERVICE" != 1 ] && { err "zero-launcher: --replace-service needs --install-service"; return 2; }
  case "$mode" in
    check) print_check; return $?;;
    list) list_models; return 0;;
    download)
      m_known "$arg_dl" || { err "zero-launcher: --download: no model \"$arg_dl\" (--list shows the keys)"; return 2; }
      model_path "$arg_dl" >/dev/null || download "$arg_dl" "$YES"; return $?;;
  esac
  if ! run_check; then
    say "missing:"; printf '%s\n' "${CHECK_LINES[@]}" | grep MISS | sed 's/^  MISS  /  /'
    say "(./zero-launcher.sh --check lists everything)"
    [ -x "$SERVERZ" ] || return 1
  fi
  if [ -n "$arg_model" ]; then
    if m_known "$arg_model"; then start_key "$arg_model"; return $?; fi
    [ -f "$arg_model" ] || { err "zero-launcher: --model: no catalog model or file \"$arg_model\" (--list shows the keys)"; return 2; }
    server_cmd "$(readlink -f "$arg_model")" "$MTP"
    if [ "$INSTALL_SERVICE" = 1 ]; then install_service; else run_server; fi
    return $?
  fi
  [ "$INSTALL_SERVICE" = 1 ] && { err "zero-launcher: --install-service needs --model"; return 2; }
  menu
}

# (sourced by tests/test_zero_launcher.sh for its functions; run as a program otherwise)
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
  exit $?
fi
