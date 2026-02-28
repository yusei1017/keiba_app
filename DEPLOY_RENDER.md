# 外部公開手順（Render）

この手順で、Wi-Fiが違う場所からでも使える公開URLを作れます。

## 1. GitHubへpush

このフォルダをGitHubリポジトリにpushします。  
`black-practice-...json` のような秘密鍵ファイルはpushしないでください。

## 2. Renderでデプロイ

1. Renderにログイン
2. `New +` → `Web Service`
3. GitHubリポジトリを接続
4. `Blueprint` を選ぶ（`render.yaml` を自動読込）
5. Deploy

完了すると `https://xxxx.onrender.com` のURLが発行されます。  
このURLをスマホで開けば利用できます。

## 3. Google Sheetsを使う場合（任意）

Renderのサービス設定で Environment Variables を追加:

- `GOOGLE_SERVICE_ACCOUNT_JSON`
  - サービスアカウントJSONの中身をそのまま1行文字列で貼る
  - または `GOOGLE_SERVICE_ACCOUNT_JSON_B64` に base64 文字列を設定

アプリ側は環境変数から自動で読み取れるようになっています。

## 4. 注意

- Freeプランはスリープ復帰に時間がかかる場合があります。
- netkeiba側の一時ブロックやレート制限時は取得に失敗することがあります。
