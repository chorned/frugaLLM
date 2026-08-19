import asyncio
import httpx
import time

async def main():
    start = time.time()
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "http://127.0.0.1:4000/v1/chat/completions",
                json={
                    "model": "frugallm",
                    "messages": [{"role": "user", "content": "Write a 5 page essay."}]
                },
                headers={"Authorization": "Bearer sk-sidecar-1"},
                timeout=1800.0  # we want to see if litellm times out on its own
            )
            print(f"Status: {resp.status_code}")
            print(f"Time: {time.time() - start:.2f}s")
            print(resp.text[:200])
    except Exception as e:
        print(f"Exception: {e}")
        print(f"Time: {time.time() - start:.2f}s")

asyncio.run(main())
