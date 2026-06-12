#!/usr/bin/env bash
set -euo pipefail

ENABLE_HYBRID=false

# Parse optional flags before positional args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --enable-hybrid) ENABLE_HYBRID=true; shift ;;
        --) shift; break ;;
        -*) echo "Unknown flag: $1" >&2; exit 1 ;;
        *) break ;;
    esac
done

if [ $# -lt 2 ]; then
    echo "Usage: $0 [--enable-hybrid] <hetzner-host-ip> <git-tag>" >&2
    echo "REPO_URL env var must be set if /opt/tradingagents/repo doesn't already exist on the host." >&2
    exit 1
fi

HOST=$1
TAG=$2
SSH="ssh tabot@$HOST"
SSH_ROOT="ssh root@$HOST"

REPO_URL="${REPO_URL:-}"

echo "→ cloning or updating repo"
if [ -n "$REPO_URL" ]; then
    $SSH "[ -d /opt/tradingagents/repo ] || git clone $REPO_URL /opt/tradingagents/repo"
else
    $SSH "[ -d /opt/tradingagents/repo ] || (echo 'ERROR: /opt/tradingagents/repo missing and REPO_URL not set'; exit 1)"
fi
$SSH "cd /opt/tradingagents/repo && git fetch --tags && git checkout $TAG"

echo "→ creating venv + installing"
$SSH "[ -d /opt/tradingagents/venv ] || python3.12 -m venv /opt/tradingagents/venv"
$SSH "/opt/tradingagents/venv/bin/pip install -U pip wheel && /opt/tradingagents/venv/bin/pip install -e /opt/tradingagents/repo"

echo "→ ensuring data + log dirs"
$SSH "mkdir -p /opt/tradingagents/data /opt/tradingagents/data-hybrid /opt/tradingagents/logs /opt/tradingagents/secrets && chmod 700 /opt/tradingagents/secrets"

echo "→ checking secrets file exists"
$SSH "[ -f /opt/tradingagents/secrets/.env.trading ] || (echo 'ERROR: scp secrets/.env.trading manually before re-running'; exit 1)"
$SSH "chmod 600 /opt/tradingagents/secrets/.env.trading"

echo "→ checking monitor secrets file exists"
$SSH "[ -f /opt/tradingagents/secrets/.env.monitor ] || (echo 'WARNING: /opt/tradingagents/secrets/.env.monitor missing — create it with TA_MONITOR_PASSWORD before monitor UI starts'; true)"

echo "→ installing systemd units (root)"
$SSH_ROOT "cp /opt/tradingagents/repo/deploy/systemd/*.service /etc/systemd/system/"
$SSH_ROOT "cp /opt/tradingagents/repo/deploy/systemd/*.timer /etc/systemd/system/"
$SSH_ROOT "systemctl daemon-reload"
$SSH_ROOT "systemctl enable --now ta-cycle.timer ta-rebacktest.timer"
$SSH_ROOT "systemctl enable --now ta-monitor.service"

echo "→ verifying timers"
$SSH_ROOT "systemctl list-timers ta-cycle.timer ta-rebacktest.timer --no-pager"

echo "→ verifying monitor UI service"
$SSH_ROOT "systemctl is-active ta-monitor.service || true"
echo "    monitor UI running on 127.0.0.1:8800 (reverse-proxy terminates TLS)"
echo "    NOTE: create /opt/tradingagents/secrets/.env.monitor with"
echo "          TA_MONITOR_PASSWORD before first start, and install Caddy"
echo "          with deploy/Caddyfile for public HTTPS access."

echo "→ installing hybrid systemd units (root)"
$SSH_ROOT "cp /opt/tradingagents/repo/deploy/systemd/ta-hybrid-cycle.service /etc/systemd/system/"
$SSH_ROOT "cp /opt/tradingagents/repo/deploy/systemd/ta-hybrid-cycle.timer /etc/systemd/system/"
$SSH_ROOT "systemctl daemon-reload"

if [ "$ENABLE_HYBRID" = true ]; then
    echo "→ enabling hybrid timer (--enable-hybrid set)"
    $SSH_ROOT "systemctl enable --now ta-hybrid-cycle.timer"
    $SSH_ROOT "systemctl list-timers ta-hybrid-cycle.timer --no-pager"
else
    echo "→ hybrid timer NOT enabled (re-run with --enable-hybrid after dry-run passes)"
fi

echo "✓ deploy complete; pinned tag: $TAG"
