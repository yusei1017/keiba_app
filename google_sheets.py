from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class GoogleSheetsUnavailable(RuntimeError):
    pass


def _sanitize_sheet_name(name: str) -> str:
    # Google Sheets sheet title restrictions: can't contain []:*?/\\ and max 100 chars.
    s = str(name or "").strip()
    if not s:
        return "sheet"
    for ch in "[]:*?/\\":  # noqa: ISC003
        s = s.replace(ch, "_")
    return s[:100]


def ensure_sheet_exists(
    *,
    spreadsheet_id: str,
    sheet_name: str,
    service_account_info: Dict[str, Any],
) -> None:
    """
    指定のシート（タブ）が無ければ作成する。
    """
    try:
        from google.oauth2.service_account import Credentials  # type: ignore
        from googleapiclient.discovery import build  # type: ignore
    except Exception as e:  # pragma: no cover
        raise GoogleSheetsUnavailable(
            "Google Sheets 連携に必要なライブラリがありません。"
            " `pip install google-api-python-client google-auth` を入れてください。"
        ) from e

    title = _sanitize_sheet_name(sheet_name)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
    service = build("sheets", "v4", credentials=creds, cache_discovery=False)

    meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    sheets = meta.get("sheets", []) or []
    for s in sheets:
        props = s.get("properties") or {}
        if str(props.get("title", "")) == title:
            return

    body = {"requests": [{"addSheet": {"properties": {"title": title}}}]}
    service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=body).execute()


def append_row(
    *,
    spreadsheet_id: str,
    sheet_name: str,
    values: List[Any],
    service_account_info: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Google Sheets に 1行追加します（service account）。

    依存:
      - google-api-python-client
      - google-auth
    """
    try:
        from google.oauth2.service_account import Credentials  # type: ignore
        from googleapiclient.discovery import build  # type: ignore
    except Exception as e:  # pragma: no cover
        raise GoogleSheetsUnavailable(
            "Google Sheets 連携に必要なライブラリがありません。"
            " `pip install google-api-python-client google-auth` を入れてください。"
        ) from e

    if not spreadsheet_id:
        raise ValueError("spreadsheet_id is required.")
    if not sheet_name:
        raise ValueError("sheet_name is required.")

    sheet_name = _sanitize_sheet_name(sheet_name)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)

    service = build("sheets", "v4", credentials=creds, cache_discovery=False)
    body = {"values": [values]}
    result = (
        service.spreadsheets()
        .values()
        .append(
            spreadsheetId=spreadsheet_id,
            range=f"{sheet_name}!A1",
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body=body,
        )
        .execute()
    )
    return json.loads(json.dumps(result))


def append_rows(
    *,
    spreadsheet_id: str,
    sheet_name: str,
    rows: List[List[Any]],
    service_account_info: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Google Sheets に複数行をまとめて追加します（service account）。
    """
    if not rows:
        return None
    try:
        from google.oauth2.service_account import Credentials  # type: ignore
        from googleapiclient.discovery import build  # type: ignore
    except Exception as e:  # pragma: no cover
        raise GoogleSheetsUnavailable(
            "Google Sheets 連携に必要なライブラリがありません。"
            " `pip install google-api-python-client google-auth` を入れてください。"
        ) from e

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
    service = build("sheets", "v4", credentials=creds, cache_discovery=False)
    sheet_name = _sanitize_sheet_name(sheet_name)
    body = {"values": rows}
    result = (
        service.spreadsheets()
        .values()
        .append(
            spreadsheetId=spreadsheet_id,
            range=f"{sheet_name}!A1",
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body=body,
        )
        .execute()
    )
    return json.loads(json.dumps(result))
