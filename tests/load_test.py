# tests/load_test.py
import asyncio
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor

async def load_test():
    """Test de carga para Neural Nexus"""
    
    async def make_request(session, url, data):
        async with session.post(url, json=data) as response:
            return await response.json()
    
    url = "http://localhost:8080/inference"
    test_data = {"model": "test", "data": list(range(100))}
    
    # 1000 requests concurrentes
    async with aiohttp.ClientSession() as session:
        tasks = [
            make_request(session, url, test_data) 
            for _ in range(1000)
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        print(f"Completed 1000 requests in {end_time - start_time:.2f}s")
        print(f"RPS: {1000 / (end_time - start_time):.2f}")

if __name__ == "__main__":
    asyncio.run(load_test())
