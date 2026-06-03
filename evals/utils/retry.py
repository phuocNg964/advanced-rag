import asyncio
import math
from src.core.logger import get_logger

logger = get_logger(__name__)

async def eval_with_retry(func, max_retries=5, **kwargs):
    """Call a RAGAS async scorer with exponential backoff on rate limits."""
    for attempt in range(max_retries):
        try:
            result = await func(**kwargs)
            val = getattr(result, 'value', None)
            if val is None:
                try:
                    val = float(result)
                except (ValueError, TypeError):
                    return None
            if val is not None and not math.isnan(val):
                return round(float(val), 4)
            return None
        except Exception as e:
            err_str = str(e)
            if any(k in err_str for k in ('429', 'Rate limit', '502', 'Timeout')):
                wait_time = (attempt + 1) * 10.0
                logger.warning(f"[Eval API] Rate limit. Waiting {wait_time:.1f}s... (Attempt {attempt+1})")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Eval error: {e}")
                return None
    return None
