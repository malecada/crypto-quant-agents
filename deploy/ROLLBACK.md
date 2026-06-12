# Rollback procedure

## Halt only (no data change)

```bash
ssh tabot@<host>
sudo systemctl stop ta-cycle.timer ta-rebacktest.timer
```

Cycles will not run again until timers are re-enabled.

## Halt + close all open positions

```bash
ssh tabot@<host>
sudo systemctl stop ta-cycle.timer ta-rebacktest.timer
cd /opt/tradingagents/repo
set -a; source /opt/tradingagents/secrets/.env.trading; set +a
/opt/tradingagents/venv/bin/python -m tradingagents.execution.live.runner --kill-all
```

## Weekly rebacktest is intentionally not enabled on first deploy

`ta-rebacktest.timer` ships with the systemd units but `deploy.sh` does
not enable it on first boot. The pred-dir input is currently hardcoded
in `rebacktest.compute_backtest_metrics` (`BACKTEST_PRED_DIR`) and the
weekly-comparison semantics need design work before turning it on. To
re-enable manually once that work lands:

```bash
sudo systemctl enable --now ta-rebacktest.timer
```

## Roll back to previous git tag

```bash
ssh tabot@<host>
sudo systemctl stop ta-cycle.timer
cd /opt/tradingagents/repo
git fetch --tags
git checkout <previous-tag>
/opt/tradingagents/venv/bin/pip install -e /opt/tradingagents/repo
sudo systemctl start ta-cycle.timer
```

## Restore data from snapshot

```bash
# On Hetzner Cloud Console: select "Snapshots" → restore latest
# This replaces the entire VM. After restore, re-run /opt/tradingagents/repo/deploy/deploy.sh
```

## Rebuild from scratch

```bash
# Locally:
./deploy/provision_hetzner.sh <new-host-ip>
scp /path/to/.env.trading tabot@<new-host>:/opt/tradingagents/secrets/.env.trading
./deploy/deploy.sh <new-host-ip> <git-tag>
```

## V5 → V1 emergency rollback

If V5 (live-v2.0) misbehaves after deploy, restore live-v1.0 in ~3 minutes.

```bash
# 1. Stop timers
systemctl stop ta-cycle.timer ta-rebacktest.timer

# 2. Kill any open positions under V5
set -a; source /opt/tradingagents/secrets/.env.trading; set +a
/opt/tradingagents/venv/bin/python -m tradingagents.execution.live.runner --kill-all

# 3. Revert code to live-v1.0
cd /opt/tradingagents
sudo -u tabot git checkout live-v1.0
sudo -u tabot /opt/tradingagents/venv/bin/pip install -e .

# 4. Restore env (drop V5 keys, restore V1 kelly)
sudo -u tabot sed -i '/^COINGLASS_API_KEY=/d; /^COIN_UNIVERSE=/d; /^KELLY_FRACTION=/d' \
    /opt/tradingagents/secrets/.env.trading
sudo -u tabot bash -c 'echo "KELLY_FRACTION=0.33" >> /opt/tradingagents/secrets/.env.trading'

# 5. Schema rollback unnecessary — V5 columns are additive, V1 ignores them.

# 6. Restart timers
systemctl start ta-cycle.timer ta-rebacktest.timer
systemctl status ta-cycle.timer
```

Journal backup from the deploy step lives at `/root/backup_pre_v5_YYYYMMDD.tar.gz`.
Restore only if SQLite corruption suspected:
```bash
systemctl stop ta-cycle.timer
tar xzf /root/backup_pre_v5_YYYYMMDD.tar.gz -C /
systemctl start ta-cycle.timer
```
