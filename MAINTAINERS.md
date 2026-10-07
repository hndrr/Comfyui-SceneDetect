# メンテナンス運用

## Comfy Registry への公開

公開設定は [publish_action.yml](.github/workflows/publish_action.yml) にあります。`master` への push・PR マージで自動公開が起動します。

通常は `pyproject.toml` のパッチ番号を自動で上げ、バージョン更新を `master` にコミットしてから公開します。例：`1.2.0` → `1.2.1`。GitHub Actions が作る準備コミットでは、新しい公開処理は起動しません。

| 変更 | バージョンの指定 |
|---|---|
| 通常の修正 | パッチ番号を自動更新 |
| 新機能の追加 | PR で `project.version` を `1.3.0` などに明示的に上げる |
| 互換性を壊す変更 | PR で `project.version` を `2.0.0` などに明示的に上げる |

明示的に上げた番号は、そのまま公開します。バージョンを下げた場合はエラーになります。

公開処理はキューで待機し、1件ずつ実行します。短時間に複数のマージがあると、最新の `master` を使った1つのリリースにまとめる場合があります。準備コミットの `Registry-Source: <40桁のコミットSHA>` で対象の変更を記録し、同じ変更を含む後続の実行はスキップします。明示的なバージョン更新でも、記録のための空コミットを作ります。

## 公開失敗時の再試行

1. GitHub の **Actions** で **Publish to Comfy registry** を開く。
2. **Run workflow** を選び、ブランチを `master` にする。
3. **Run workflow** で実行し、**Publish Custom Node** ステップの結果を確認する。

手動実行は現在のバージョンを公開し、番号を上げません。

準備コミットの push 後に公開が失敗した場合、**Re-run** では準備済みと判定され、公開をスキップしたまま成功扱いになることがあります。公開の再試行には **Run workflow** を使います。

## 公開設定の管理

- Registry の認証情報はリポジトリの Actions secret `REGISTRY_ACCESS_TOKEN` に保存します。
- バージョンの準備には `contents: write` 権限と `master` への push が必要です。Git の書き込み認証情報は公開 Action の実行前に削除します。
- 公開 Action はコミット `d2366e7abb6ab16f3bb03e3520ae25c8cf749bc9` に固定しています。Action 本体を更新するときは、更新先の内容を確認してワークフローの SHA を書き換え、コミット・PR に含めます。
- 公開 Action の `skip_checkout: 'true'` は、準備したバージョンを公開に使うために必要です。
