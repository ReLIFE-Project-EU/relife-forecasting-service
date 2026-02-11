"""Tests for retry utilities."""

import asyncio
import pytest

from relife_forecasting.utils.retry import retry_on_transient_error, TRANSIENT_EXCEPTIONS


def test_retry_successful_on_first_attempt():
    """Test that a successful function call on first attempt works."""
    call_count = [0]

    @retry_on_transient_error()
    def successful_function():
        call_count[0] += 1
        return "success"

    result = successful_function()
    assert result == "success"
    assert call_count[0] == 1


def test_retry_with_transient_error_then_success():
    """Test that function retries on transient errors until success."""
    call_count = [0]

    @retry_on_transient_error(max_retries=3, initial_delay=0.01)
    def flaky_function():
        call_count[0] += 1
        if call_count[0] < 3:
            raise ConnectionError("Network issue")
        return "success"

    result = flaky_function()
    assert result == "success"
    assert call_count[0] == 3


def test_retry_exhausts_retries():
    """Test that function raises exception after exhausting retries."""
    call_count = [0]

    @retry_on_transient_error(max_retries=3, initial_delay=0.01)
    def always_fails():
        call_count[0] += 1
        raise ConnectionError("Network issue")

    with pytest.raises(ConnectionError):
        always_fails()
    
    assert call_count[0] == 3


def test_retry_non_transient_exception_not_retried():
    """Test that non-transient exceptions are not retried."""
    call_count = [0]

    @retry_on_transient_error(max_retries=3, initial_delay=0.01)
    def raises_value_error():
        call_count[0] += 1
        raise ValueError("Not a transient error")

    with pytest.raises(ValueError):
        raises_value_error()
    
    assert call_count[0] == 1


def test_retry_parameter_validation():
    """Test that invalid parameters raise ValueError."""
    with pytest.raises(ValueError, match="max_retries must be >= 1"):
        retry_on_transient_error(max_retries=0)

    with pytest.raises(ValueError, match="initial_delay must be >= 0"):
        retry_on_transient_error(initial_delay=-1)

    with pytest.raises(ValueError, match="backoff_factor must be > 0"):
        retry_on_transient_error(backoff_factor=0)


@pytest.mark.asyncio
async def test_async_retry_successful():
    """Test that async functions are properly handled."""
    call_count = [0]

    @retry_on_transient_error()
    async def async_successful_function():
        call_count[0] += 1
        await asyncio.sleep(0.01)
        return "async success"

    result = await async_successful_function()
    assert result == "async success"
    assert call_count[0] == 1


@pytest.mark.asyncio
async def test_async_retry_with_transient_error():
    """Test that async function retries on transient errors."""
    call_count = [0]

    @retry_on_transient_error(max_retries=3, initial_delay=0.01)
    async def async_flaky_function():
        call_count[0] += 1
        await asyncio.sleep(0.01)
        if call_count[0] < 3:
            raise TimeoutError("Timeout")
        return "async success"

    result = await async_flaky_function()
    assert result == "async success"
    assert call_count[0] == 3


def test_requests_exception_is_transient():
    """Test that requests exceptions are included in transient exceptions."""
    try:
        from requests.exceptions import RequestException
        assert RequestException in TRANSIENT_EXCEPTIONS or any(
            issubclass(RequestException, exc) for exc in TRANSIENT_EXCEPTIONS
        )
    except ImportError:
        pytest.skip("requests library not installed")
