import re
import json
from typing import List, Literal, Optional, Union

from selectolax.lexbor import LexborHTMLParser, LexborNode

from .decoder import DecodedResult, ResultDecoder
from .schema import Flight, Result
from .flights_impl import FlightData, Passengers
from .filter import TFSData
from .fallback_playwright import fallback_playwright_fetch
from .bright_data_fetch import bright_data_fetch
from .primp import Client, Response


DataSource = Literal['html', 'js']

# Default cookies embedded into the app to help bypass common consent gating.
# These are used only if the caller does not supply cookies (binary) and
# does not provide cookies via request_kwargs.
_DEFAULT_COOKIES = {
    "CONSENT": "YES+cb.20210720-07-p0.en+FX+410",
    "SOCS": "CAESHAgBEhJnd3NfMjAyMzA4MTAtMF9SQzIaAmRlIAEaBgiAo_CmBg",
}
_DEFAULT_COOKIES_BYTES = json.dumps(_DEFAULT_COOKIES).encode("utf-8")


def fetch(params: dict, request_kwargs: dict | None = None) -> Response:
    req_kwargs = request_kwargs.copy() if request_kwargs else {}
    # proxy is a Client constructor param, not a get() param — extract it
    proxy = req_kwargs.pop('proxy', None)
    # _trace_writer is optional — pop before passing to HTTP client
    _tw = req_kwargs.pop('_trace_writer', None)

    base_url = "https://www.google.com/travel/flights"
    from urllib.parse import urlencode
    full_url = f"{base_url}?{urlencode(params)}"

    import datetime as _dt
    if _tw:
        _tw.write(
            f"\n  >> HTTP GET {full_url}\n"
            f"     Proxy  : {proxy or 'none'}\n"
        )

    client = Client(impersonate="chrome_126", verify=False, proxy=proxy)
    res = client.get(base_url, params=params, **req_kwargs)

    if _tw:
        body_preview = (res.text or '')[:300].replace('\n', ' ')
        _tw.write(
            f"     Status : {res.status_code}   Size : {len(res.text or '')} chars\n"
            f"     Body   : {body_preview}{'…' if len(res.text or '') > 300 else ''}\n"
        )

    assert res.status_code == 200, f"{res.status_code} Result: {res.text_markdown}"
    return res


def _merge_binary_cookies(cookies_bytes: bytes | None, request_kwargs: dict | None) -> dict:
    """Parse binary cookies into request kwargs.

    Supported formats (in order):
    - JSON bytes -> dict or list of pairs
    - Pickle bytes -> dict
    - Raw cookie header bytes -> sets the 'Cookie' header

    Existing request_kwargs are copied and updated; existing 'cookies' or 'headers' are overridden by parsed values.
    """
    req_kwargs = request_kwargs.copy() if request_kwargs else {}
    if not cookies_bytes:
        return req_kwargs

    # Try JSON first
    try:
        s = cookies_bytes.decode("utf-8")
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            req_kwargs['cookies'] = parsed
            return req_kwargs
        if isinstance(parsed, list):
            # list of pairs
            try:
                req_kwargs['cookies'] = dict(parsed)
                return req_kwargs
            except Exception:
                pass
    except Exception:
        pass

    # Try pickle
    try:
        import pickle

        parsed = pickle.loads(cookies_bytes)
        if isinstance(parsed, dict):
            req_kwargs['cookies'] = parsed
            return req_kwargs
    except Exception:
        pass

    # Fallback: treat as raw Cookie header
    try:
        s = cookies_bytes.decode("utf-8")
        headers = req_kwargs.get('headers', {})
        # make a shallow copy to avoid mutating input
        headers = headers.copy() if isinstance(headers, dict) else {}
        headers['Cookie'] = s
        req_kwargs['headers'] = headers
    except Exception:
        # give up silently and return what we have
        pass

    return req_kwargs


def get_flights_from_filter(
    filter: TFSData,
    currency: str = "",
    *,
    mode: Literal["common", "fallback", "force-fallback", "local", "bright-data"] = "common",
    data_source: DataSource = 'html',
    cookies: bytes | None = None,
    request_kwargs: dict | None = None,
    cookie_consent: bool = True,
) -> Union[Result, DecodedResult, None]:
    data = filter.as_b64()

    params = {
        "tfs": data.decode("utf-8"),
        "hl": "en",
        "tfu": "EgQIABABIgA",
        "curr": currency,
    }

    # If the caller didn't provide cookies bytes and there is no cookies or Cookie header
    # in request_kwargs, use the embedded default cookies bytes (only when enabled).
    if cookies is None and cookie_consent:
        has_cookies_in_req = False
        if request_kwargs:
            if 'cookies' in request_kwargs:
                has_cookies_in_req = True
            elif 'headers' in request_kwargs and isinstance(request_kwargs['headers'], dict) and 'Cookie' in request_kwargs['headers']:
                has_cookies_in_req = True
        if not has_cookies_in_req:
            cookies = _DEFAULT_COOKIES_BYTES

    # Merge binary cookies into request kwargs (binary cookies take precedence)
    req_kwargs = _merge_binary_cookies(cookies, request_kwargs)

    if mode in {"common", "fallback"}:
        try:
            res = fetch(params, request_kwargs=req_kwargs)
        except AssertionError as e:
            if mode == "fallback":
                res = fallback_playwright_fetch(params, request_kwargs=req_kwargs)
            else:
                raise e

    elif mode == "local":
        from .local_playwright import local_playwright_fetch

        res = local_playwright_fetch(params, request_kwargs=req_kwargs)

    elif mode == "bright-data":
        res = bright_data_fetch(params, request_kwargs=req_kwargs)

    else:
        res = fallback_playwright_fetch(params, request_kwargs=req_kwargs)

    try:
        return parse_response(res, data_source)
    except RuntimeError as e:
        if mode == "fallback":
            return get_flights_from_filter(filter, mode="force-fallback", request_kwargs=req_kwargs, cookies=None, cookie_consent=cookie_consent)
        raise e



def get_flights(
    *,
    flight_data: List[FlightData],
    trip: Literal["round-trip", "one-way", "multi-city"],
    passengers: Optional[Passengers] = None,
    # Convenience passenger counters (used when `passengers` is None)
    adults: Optional[int] = None,
    children: int = 0,
    infants_in_seat: int = 0,
    infants_on_lap: int = 0,
    seat: Literal["economy", "premium-economy", "business", "first"] = "economy",
    fetch_mode: Literal["common", "fallback", "force-fallback", "local", "bright-data"] = "common",
    max_stops: Optional[int] = None,
    data_source: DataSource = 'html',
    cookies: bytes | None = None,
    request_kwargs: dict | None = None,
    cookie_consent: bool = True,
) -> Union[Result, DecodedResult, None]:
    # If the caller didn't supply a Passengers object, build one from the
    # convenience counters. Default to 1 adult when no adults count provided
    # (matches previous typical usage where at least one adult is expected).
    if passengers is None:
        ad = 1 if adults is None else adults
        passengers = Passengers(
            adults=ad,
            children=children,
            infants_in_seat=infants_in_seat,
            infants_on_lap=infants_on_lap,
        )

    tfs: TFSData = TFSData.from_interface(
        flight_data=flight_data,
        trip=trip,
        passengers=passengers,
        seat=seat,
        max_stops=max_stops,
    )

    return get_flights_from_filter(
        tfs,
        mode=fetch_mode,
        data_source=data_source,
        cookies=cookies,
        request_kwargs=request_kwargs,
        cookie_consent=cookie_consent,
    )



def parse_response(
     r: Response,
     data_source: DataSource,
     *,
     dangerously_allow_looping_last_item: bool = False,
 ) -> Union[Result, DecodedResult, None]:
    class _blank:
        def text(self, *_, **__):
            return ""

        def iter(self):
            return []

    blank = _blank()

    def safe(n: Optional[LexborNode]):
        return n or blank

    parser = LexborHTMLParser(r.text)

    if data_source == 'js':
        script = parser.css_first(r'script.ds\:1').text()

        match = re.search(r'^.*?\{.*?data:(\[.*\]).*}', script)
        assert match, 'Malformed js data, cannot find script data'
        data = json.loads(match.group(1))
        return ResultDecoder.decode(data) if data is not None else None

    flights = []

    for i, fl in enumerate(parser.css('div[jsname="IWWDBc"], div[jsname="YdtKid"]')):
        is_best_flight = i == 0

        for item in fl.css("ul.Rk10dc li")[
            : (None if dangerously_allow_looping_last_item or i == 0 else -1)
        ]:
            # Flight name
            name = safe(item.css_first("div.sSHqwe.tPgKwe.ogfYpf span")).text(
                strip=True
            )

            # Get departure & arrival time (outbound [0],[1]; return leg [2],[3] for round trips)
            dp_ar_node = item.css("span.mv1WYe div")
            try:
                departure_time = dp_ar_node[0].text(strip=True)
                arrival_time = dp_ar_node[1].text(strip=True)
            except IndexError:
                departure_time = ""
                arrival_time = ""
            return_departure_time = dp_ar_node[2].text(strip=True) if len(dp_ar_node) > 2 else ""
            return_arrival_time = dp_ar_node[3].text(strip=True) if len(dp_ar_node) > 3 else ""

            # Get arrival time ahead
            time_ahead = safe(item.css_first("span.bOzv6")).text()

            # Get duration (outbound first, return second for round trips if present)
            def _is_duration(s):
                return bool(s and re.search(r'\d+\s*(hr|min|h|m)\b', s, re.I))

            duration_nodes = item.css("li div.Ak5kof div")
            duration = duration_nodes[0].text(strip=True) if duration_nodes else ""
            _rd = duration_nodes[1].text(strip=True) if len(duration_nodes) > 1 else ""
            return_duration = _rd if _is_duration(_rd) else ""

            # Get flight stops (outbound first, return second for round trips if present)
            stops_nodes = item.css(".BbR8Ec .ogfYpf")
            stops = stops_nodes[0].text(strip=True) if stops_nodes else ""
            _rs = stops_nodes[1].text(strip=True) if len(stops_nodes) > 1 else ""
            return_stops_raw = _rs if (_rs == "Nonstop" or re.match(r'^\d+', _rs or '')) else ""

            # Get delay
            delay = safe(item.css_first(".GsCCve")).text() or None

            # Get prices
            price = safe(item.css_first(".YMlIz.FpEdX")).text() or "0"

            # Stops formatting
            def _parse_stops(s):
                try:
                    return 0 if s == "Nonstop" else int(s.split(" ", 1)[0])
                except (ValueError, AttributeError):
                    return 0

            flights.append(
                {
                    "is_best": is_best_flight,
                    "name": name,
                    "departure": " ".join(departure_time.split()),
                    "arrival": " ".join(arrival_time.split()),
                    "arrival_time_ahead": time_ahead,
                    "duration": duration,
                    "stops": _parse_stops(stops),
                    "delay": delay,
                    "price": price.replace(",", ""),
                    "return_departure": " ".join(return_departure_time.split()),
                    "return_arrival": " ".join(return_arrival_time.split()),
                    "return_duration": return_duration,
                    "return_stops": _parse_stops(return_stops_raw),
                }
            )

    current_price = safe(parser.css_first("span.gOatQ")).text()
    if not flights:
        # HTML mode found no flights. Check if this is a valid Google Flights page
        # (has ds:1 script) — if so, try JS mode as a fallback before giving up.
        # This handles cases where Google serves the page without pre-rendered HTML
        # divs but the ds:1 script still contains embedded flight data.
        ds1_el = parser.css_first(r'script.ds\:1')
        if ds1_el:
            try:
                js_result = parse_response(r, 'js')
                if js_result is not None:
                    return js_result
            except Exception:
                pass
            # ds:1 present but no data in either mode — genuine no results
            title_el = parser.css_first('title')
            title_text = title_el.text() if title_el else ''
            if 'Google Flights' in title_text:
                raise RuntimeError("No results for this route")
        raise RuntimeError("No flights found:\n{}".format(r.text_markdown))

    return Result(current_price=current_price, flights=[Flight(**fl) for fl in flights])  # type: ignore