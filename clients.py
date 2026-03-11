import httpx
import logging
import asyncio  # <--- THIS WAS MISSING OR CAUSING THE ERROR
from typing import List, Tuple, Dict, Any
from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool

# Mute noisy logs
logging.getLogger("nebula3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Dedicated logger for errors
err_log = logging.getLogger("clients_error")
err_log.setLevel(logging.ERROR)

class ThreadSafeNebulaClient:
    def __init__(self, host, port, user, password, space):
        cfg = Config()
        cfg.max_connection_pool_size = 50
        self.pool = ConnectionPool()
        if not self.pool.init([(host, port)], cfg):
            raise RuntimeError("Failed to init Nebula connection pool")
        self.user = user
        self.password = password
        self.space = space

    def _get_sess(self):
        try:
            sess = self.pool.get_session(self.user, self.password, True)
            sess.execute(f"USE {self.space}")
            return sess
        except Exception as e:
            err_log.error(f"Failed to get Nebula session: {e}")
            raise

    def lookup_semantic_vids(self, suis: List[str], limit_each: int = 1) -> Dict[str, str]:
        out = {}
        if not suis: return out
        sess = self._get_sess()
        try:
            for s in suis:
                ngql = f'LOOKUP ON Semantic WHERE Semantic.SUI == "{s}" YIELD id(vertex) AS vid | LIMIT {int(limit_each)};'
                res = sess.execute(ngql)
                if res.is_succeeded() and res.row_size() > 0:
                    out[s] = str(res.row_values(0)[0]).strip('"')
                elif not res.is_succeeded():
                    err_log.warning(f"Lookup failed for SUI {s}: {res.error_msg()}")
        finally:
            sess.release()
        return out

    def sty_reverse_concepts(self, sem_vids: List[str], limit_each: int = 200) -> Dict[str, List[str]]:
        out = {}
        if not sem_vids: return out
        sess = self._get_sess()
        try:
            for sv in sem_vids:
                ngql = f'GO FROM "{sv}" OVER STY REVERSELY YIELD DISTINCT id($$) AS concept_vid | LIMIT {int(limit_each)};'
                res = sess.execute(ngql)
                concepts = []
                if res.is_succeeded():
                    for i in range(res.row_size()):
                        concepts.append(str(res.row_values(i)[0]).strip('"'))
                out[sv] = concepts
        finally:
            sess.release()
        return out

    def def_pairs_for_concepts(self, concept_vids: List[str], limit_total: int = 6000) -> List[Tuple[str, str]]:
        if not concept_vids: return []
        sess = self._get_sess()
        try:
            # Handle empty strings or quotes safely
            quoted = ", ".join([f"\"{c}\"" for c in concept_vids if c])
            if not quoted: return []
            ngql = f'GO FROM {quoted} OVER DEF YIELD DISTINCT id($^) AS concept_vid, id($$) AS def_vid | LIMIT {int(limit_total)};'
            res = sess.execute(ngql)
            pairs = []
            if res.is_succeeded():
                for i in range(res.row_size()):
                    pairs.append((str(res.row_values(i)[0]).strip('"'), str(res.row_values(i)[1]).strip('"')))
            else:
                err_log.warning(f"Def pairs query failed: {res.error_msg()}")
        finally:
            sess.release()
        return pairs

    def fetch_def_props(self, def_vids: List[str]) -> Dict[str, Dict[str, Any]]:
        out = {}
        if not def_vids: return out
        sess = self._get_sess()
        try:
            for i in range(0, len(def_vids), 200):
                chunk = def_vids[i:i + 200]
                q = ", ".join([f'"{d}"' for d in chunk])
                ngql = f'FETCH PROP ON Definition {q} YIELD id(vertex), Definition.ATUI, Definition.DEF, Definition.SAB;'
                res = sess.execute(ngql)
                if res.is_succeeded():
                    for r in range(res.row_size()):
                        row = res.row_values(r)
                        vid = str(row[0]).strip('"')
                        out[vid] = {
                            "def_vid": vid, 
                            "ATUI": str(row[1]).strip('"'), 
                            "DEF": str(row[2]).strip('"'), 
                            "SAB": str(row[3]).strip('"')
                        }
        finally:
            sess.release()
        return out

class AsyncOllamaClient:
    def __init__(self, base_url, model, api_key):
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"}
        
        # INCREASED TIMEOUT TO 20 MINUTES (1200s) for 70B models
        self.client = httpx.AsyncClient(timeout=1200.0)

    async def chat(self, messages, temperature=0.0, max_retries=10, backoff_factor=5.0):
        payload = {"model": self.model, "messages": messages, "temperature": float(temperature)}
        
        for attempt in range(max_retries + 1):
            try:
                resp = await self.client.post(self.url, json=payload, headers=self.headers)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                # If we ran out of retries, raise the error
                if attempt == max_retries:
                    err_log.error(f"LLM request failed after {max_retries} attempts. Error: {e}")
                    raise e
                
                # Check if it's a server error (5xx) or timeout, which are worth retrying
                should_retry = False
                # Retry on Server Errors (504 Gateway Timeout, 500 Internal Error, 503 Service Unavailable)
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code >= 500:
                    should_retry = True
                # Retry on Connection/Timeout errors
                elif isinstance(e, httpx.RequestError):
                    should_retry = True
                
                if should_retry:
                    # SLOWER BACKOFF: 5s, 10s, 20s, 40s...
                    sleep_time = backoff_factor * (2 ** attempt)
                    logging.warning(f"LLM Error ({e}). Retrying in {sleep_time}s (Attempt {attempt+1}/{max_retries})...")
                    await asyncio.sleep(sleep_time)
                else:
                    # Client errors (4xx) usually shouldn't be retried
                    raise e