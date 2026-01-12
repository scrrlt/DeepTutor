# Troubleshooting

Quick solutions for common issues.

## Startup Issues

| Problem | Solution |
|:--------|:---------|
| Backend fails to start | Check Python ≥ 3.10, verify `.env` config |
| `npm: command not found` | Install Node.js: `conda install -c conda-forge nodejs` |
| Port already in use | Kill process: `lsof -i :8001` → `kill -9 <PID>` |
| Long path errors on Windows when cloning the repo or installing dependencies | Enable long paths*: Run as admin `reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f`, then restart your terminal |

* Restart your terminal and, if the issue persists, log out and back in (or restart Windows) for the registry change to fully take effect.
## Connection Issues

| Problem | Solution |
|:--------|:---------|
| Frontend can't reach backend | Verify backend at http://localhost:8001/docs |
| WebSocket fails | Check firewall, confirm `ws://localhost:8001/api/v1/...` format |
| Remote access fails | Set `NEXT_PUBLIC_API_BASE=http://your-ip:8001` in `.env` |

## Frontend Build Issues

| Problem | Solution |
|:--------|:---------|
| `next build` gets stuck or panics in a `tokio-runtime-worker` Turbopack Rust backtrace | Force webpack: `npm run build -- --webpack`. If it still fails, delete `web/.next` and retry. |

## Docker Issues

| Problem | Solution |
|:--------|:---------|
| Cloud frontend won't connect | Set `NEXT_PUBLIC_API_BASE_EXTERNAL=https://your-server:8001` |
| Wrong architecture | Check with `uname -m`: use `:latest` (AMD64) or `:latest-arm64` (ARM) |

## Knowledge Base

| Problem | Solution |
|:--------|:---------|
| Processing stuck | Check terminal logs, verify API keys |
| `uvloop.Loop` error | Run: `./scripts/extract_numbered_items.sh <kb_name>` |

## Kill Background Processes

```bash
# macOS/Linux
lsof -i :8001 && kill -9 <PID>

# Windows
netstat -ano | findstr :8001
taskkill /PID <PID> /F
```

---

📖 **Full FAQ**: [GitHub README](https://github.com/HKUDS/DeepTutor#-faq)
