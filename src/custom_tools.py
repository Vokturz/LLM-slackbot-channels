from langchain.tools import tool, DuckDuckGoSearchRun, ArxivQueryRun
from concurrent.futures import ThreadPoolExecutor
import asyncio
import subprocess
@tool
def disk_usage(query: str) -> str:
    """useful for when you need to answer questions about the disk usage."""
    output = subprocess.check_output(['df', '-h'], text=True)
    output = "This is the output of `df -h`:\n" + output
    return output

@tool
def memory_usage(query: str) -> str:
    """useful for when you need to answer questions about memory usage."""
    output = subprocess.check_output(['free', '-h'], text=True)
    output = "This is the output of `free -h`. Mem refers to RAM memory and Swap to swap memory:\n" + output
    return output

class asyncDuckDuckGoSearchRun(DuckDuckGoSearchRun):
    # max number of parallel requests
    max_workers: int = 2
    async def _arun(self,query: str) -> str:
        """Use the tool asynchronously."""
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        results = await asyncio.get_running_loop().run_in_executor(executor, self._run, query)
        return results
    
class asyncArxivQueryRun(ArxivQueryRun):
    # max number of parallel requests
    max_workers: int = 2
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        results = await asyncio.get_running_loop().run_in_executor(executor, self._run, query)
        return results
    
