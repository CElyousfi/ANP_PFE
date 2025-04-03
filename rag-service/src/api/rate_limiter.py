# rag-service/src/api/rate_limiter.py
import time
from typing import Dict, Tuple
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

class RateLimiter(BaseHTTPMiddleware):
    """
    Middleware for rate limiting API requests.
    """
    def __init__(
        self,
        app,
        limit: int = 100,
        window: int = 3600,
    ):
        """
        Initialize the rate limiter.
        
        Args:
            app: The FastAPI application
            limit: Maximum number of requests per window
            window: Time window in seconds
        """
        super().__init__(app)
        self.limit = limit
        self.window = window
        self.requests: Dict[str, Dict[str, Tuple[int, float]]] = {}  # client_id -> {endpoint -> (count, start_time)}
    
    async def dispatch(self, request: Request, call_next):
        """
        Process the request with rate limiting.
        """
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Get endpoint
        endpoint = f"{request.method} {request.url.path}"
        
        # Check if rate limit is exceeded
        if not self._allow_request(client_id, endpoint):
            return Response(
                content="Rate limit exceeded. Try again later.",
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                headers={"Retry-After": str(self.window)}
            )
        
        # Process the request
        response = await call_next(request)
        
        # Add rate limit headers
        client_data = self.requests.get(client_id, {})
        endpoint_data = client_data.get(endpoint, (0, 0))
        remaining = max(0, self.limit - endpoint_data[0])
        
        response.headers["X-RateLimit-Limit"] = str(self.limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(endpoint_data[1] + self.window))
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """
        Get a unique identifier for the client.
        """
        # Try API key first
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if api_key:
            return f"api_key:{api_key}"
        
        # Fall back to IP address
        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"
    
    def _allow_request(self, client_id: str, endpoint: str) -> bool:
        """
        Check if a request is allowed based on rate limits.
        """
        current_time = time.time()
        
        # Clean up expired entries
        self._cleanup()
        
        # Initialize client data
        if client_id not in self.requests:
            self.requests[client_id] = {}
        
        # Get current count and start time
        client_data = self.requests[client_id]
        count, start_time = client_data.get(endpoint, (0, current_time))
        
        # Check if window has expired
        if current_time - start_time > self.window:
            # Reset window
            client_data[endpoint] = (1, current_time)
            return True
        
        # Check if limit is exceeded
        if count >= self.limit:
            return False
        
        # Increment counter
        client_data[endpoint] = (count + 1, start_time)
        return True
    
    def _cleanup(self):
        """
        Clean up expired rate limit entries.
        """
        current_time = time.time()
        
        for client_id in list(self.requests.keys()):
            client_data = self.requests[client_id]
            
            for endpoint in list(client_data.keys()):
                count, start_time = client_data[endpoint]
                
                if current_time - start_time > self.window:
                    del client_data[endpoint]
            
            if not client_data:
                del self.requests[client_id]