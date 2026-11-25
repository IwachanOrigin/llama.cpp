#!/usr/bin/env bash

set -euo pipefail

BUILD_DIR="build-lan"
DEPLOY_DIR="$BUILD_DIR/deploy"
EXECUTABLE="qwen3moe-stage-tcp.exe"

mkdir -p "$DEPLOY_DIR"

cp "$BUILD_DIR/bin/$EXECUTABLE" "$DEPLOY_DIR/"

declare -A processed

collect_dependencies() {
    local binary="$1"
    local dependencies
    local dll
    local source
    local key

    dependencies=$(objdump -p "$binary" |
        awk '/DLL Name:/ {print $3}' |
        tr -d '\r')

    while IFS= read -r dll; do
        [[ -z "$dll" ]] && continue

        key="${dll,,}"

        [[ -n "${processed[$key]:-}" ]] && continue
        processed[$key]=1

        # Windows標準DLLは配布しない
        if [[ -f "/c/Windows/System32/$dll" ]]; then
            continue
        fi

        case "$key" in
            api-ms-win-*|ext-ms-win-*)
                continue
                ;;
        esac

        # ビルド成果物から検索
        source=$(find "$BUILD_DIR" \
            -path "$DEPLOY_DIR" -prune -o \
            -type f -iname "$dll" -print -quit)

        # UCRT64ランタイムから検索
        if [[ -z "$source" ]]; then
            source=$(find /ucrt64/bin \
                -maxdepth 1 \
                -type f -iname "$dll" -print -quit)
        fi

        if [[ -z "$source" ]]; then
            echo "ERROR: Dependency not found: $dll" >&2
            return 1
        fi

        echo "Copy: $source"

        cp "$source" "$DEPLOY_DIR/$dll"

        # DLL自身の依存関係も再帰的に調査
        collect_dependencies "$source"

    done <<< "$dependencies"
}

collect_dependencies "$DEPLOY_DIR/$EXECUTABLE"

(
    cd "$DEPLOY_DIR"
    shopt -s nullglob

    sha256sum "$EXECUTABLE" *.dll > binary-sha256.txt
)

echo
echo "Deployment package:"
ls -lh "$DEPLOY_DIR"