#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <hetzner-host-ip>" >&2
    exit 1
fi

HOST=$1
SSH="ssh -o StrictHostKeyChecking=accept-new root@$HOST"

echo "→ disabling root password auth"
$SSH "sed -i 's/^#PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config && systemctl reload ssh"

echo "→ installing UFW + fail2ban + unattended-upgrades"
$SSH "apt-get update && apt-get install -y ufw fail2ban unattended-upgrades"

echo "→ configuring UFW (ssh only)"
$SSH "ufw default deny incoming && ufw default allow outgoing && ufw allow 22/tcp && ufw --force enable"

echo "→ enabling unattended-upgrades (security only)"
$SSH "dpkg-reconfigure -f noninteractive unattended-upgrades"

echo "→ creating tabot user"
$SSH "id -u tabot >/dev/null 2>&1 || useradd -m -s /bin/bash tabot"
$SSH "mkdir -p /home/tabot/.ssh && chmod 700 /home/tabot/.ssh"

echo "→ copying SSH key from root to tabot"
$SSH "cp /root/.ssh/authorized_keys /home/tabot/.ssh/ && chown -R tabot:tabot /home/tabot/.ssh && chmod 600 /home/tabot/.ssh/authorized_keys"

echo "→ installing python3.12 + git + sqlite3"
$SSH "apt-get install -y python3.12 python3.12-venv python3.12-dev git sqlite3 build-essential"

echo "→ creating /opt/tradingagents"
$SSH "mkdir -p /opt/tradingagents && chown tabot:tabot /opt/tradingagents"

echo "✓ provisioning complete"
