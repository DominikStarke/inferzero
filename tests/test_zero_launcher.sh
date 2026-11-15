#!/usr/bin/env bash
# The Linux launcher's service writes and its refusal before a download: in a temporary folder, no GPU, no network,
# no systemctl. Run by CI.
#   bash tests/test_zero_launcher.sh
set -u
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=../zero-launcher.sh
source "$root/zero-launcher.sh"
pass=0 fail=0
ok() { if [ "$1" -eq 0 ]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL: $2"; fi; }

setup() {  # a fresh config folder and the launcher's defaults for --install-service
  T=$(mktemp -d)
  export XDG_CONFIG_HOME="$T"
  unset HF_HOME HF_HUB_CACHE XDG_CACHE_HOME HF_TOKEN_PATH
  SERVICE_NAME=inferzero REPLACE_SERVICE=0 HOST=127.0.0.1 INSTALL_SERVICE=1
  UNIT="$T/systemd/user/inferzero.service"
  CMD=(/usr/bin/serverz --model /models/a.gguf)
}
teardown() { rm -rf "$T"; }
install() { OUT=$(install_service 2>&1); return $?; }
no_temp() { [ -z "$(find "$(dirname "$UNIT")" -name '.inferzero-*' 2>/dev/null)" ]; }

# 1. a fresh unit is written, no temporary file left
setup; install; rc=$?
grep -q 'ExecStart="/usr/bin/serverz"' "$UNIT" && [ $rc -eq 0 ] && no_temp; ok $? "fresh unit"; teardown
# 2. an existing unit is left unchanged, and the message names --replace-service
setup; mkdir -p "$(dirname "$UNIT")"; printf 'private Pi unit\n' > "$UNIT"; install; rc=$?
[ $rc -eq 1 ] && [ "$(cat "$UNIT")" = "private Pi unit" ] && [[ "$OUT" == *--replace-service* ]]; ok $? "existing unit unchanged"; teardown
# 3. --replace-service replaces it
setup; install; REPLACE_SERVICE=1; CMD=(/usr/bin/other); install; rc=$?
[ $rc -eq 0 ] && grep -q 'ExecStart="/usr/bin/other"' "$UNIT"; ok $? "explicit replacement"; teardown
# 4. replacing a symlinked unit replaces the link itself, never writes through it
setup; printf 'keep me' > "$T/private.txt"; mkdir -p "$(dirname "$UNIT")"; ln -s "$T/private.txt" "$UNIT"
install; rc1=$?; REPLACE_SERVICE=1; install; rc2=$?
[ $rc1 -eq 1 ] && [ $rc2 -eq 0 ] && [ ! -L "$UNIT" ] && [ "$(cat "$T/private.txt")" = "keep me" ]; ok $? "symlink not followed"; teardown
# 5. a separate name: its own unit and printed commands
setup; SERVICE_NAME=inferzero-package-test.service; install; rc=$?
[ $rc -eq 0 ] && [ ! -e "$UNIT" ] && [ -f "$T/systemd/user/inferzero-package-test.service" ] && [[ "$OUT" == *"restart inferzero-package-test"* ]]
ok $? "separate name"; teardown
# 6. names that are not plain unit names are refused, nothing created
setup; r=0
for n in '' '../pi' '/tmp/pi' $'pi\n' 'pi unit' '-pi'; do SERVICE_NAME=$n; install && r=1; done
[ $r -eq 0 ] && [ ! -e "$T/systemd" ]; ok $? "bad names refused"; teardown
# 7. a line break in the command is refused
setup; CMD=($'/bin/server\nExecStartPre=/bin/false'); install; rc=$?
[ $rc -eq 1 ] && [ ! -e "$UNIT" ]; ok $? "line break in command"; teardown
# 8. a line break in a cache variable is refused
setup; export HF_HOME=$'/cache\nExecStartPre=/bin/false'; install; rc=$?
[ $rc -eq 1 ] && [ ! -e "$UNIT" ]; ok $? "line break in cache variable"; unset HF_HOME; teardown
# 9. systemd's specifiers and variables escaped in ExecStart=
setup; CMD=(/bin/serverz 'a$b%"c\d'); install; rc=$?
[ $rc -eq 0 ] && grep -qF '"a$$b%%\"c\\d"' "$UNIT"; ok $? "ExecStart escaping"; teardown
# 10. a unit that appears while the unit is written is kept (the create never replaces a file)
setup; mkdir -p "$(dirname "$UNIT")"
ln() { printf 'other writer' > "$UNIT"; return 1; }  # (another writer wins the race)
install; rc=$?; unset -f ln
[ $rc -eq 1 ] && [ "$(cat "$UNIT")" = "other writer" ] && no_temp; ok $? "racing writer kept"; teardown
# 11. Environment= keeps $ (systemd expands it in ExecStart= only), % doubled
setup; export HF_HOME='/cache/$x%y'; install; rc=$?
[ $rc -eq 0 ] && grep -qF 'Environment="HF_HOME=/cache/$x%%y"' "$UNIT"; ok $? "Environment keeps \$"; unset HF_HOME; teardown
# 12. a file system without hard links: the exclusive create writes the unit, no temporary file left
setup; ln() { return 1; }
install; rc=$?; unset -f ln
[ $rc -eq 0 ] && grep -q 'ExecStart="/usr/bin/serverz"' "$UNIT" && no_temp; ok $? "no hard links"; teardown
# 13. --install-service with an existing unit refuses before any download
setup; mkdir -p "$(dirname "$UNIT")"; printf 'private Pi unit\n' > "$UNIT"
model_path() { return 1; }
download() { echo "DOWNLOADED"; return 0; }
OUT=$(start_key 4b 2>&1); rc=$?
unset -f model_path download; source "$root/zero-launcher.sh"
[ $rc -eq 1 ] && [[ "$OUT" != *DOWNLOADED* ]] && [ "$(cat "$UNIT")" = "private Pi unit" ]; ok $? "refused before download"; teardown

echo "test_zero_launcher: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
