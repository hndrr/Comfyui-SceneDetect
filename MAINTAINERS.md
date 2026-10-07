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

先に、対象の準備コミット（`Prepare registry version ...`）が `master` に入っているか確認します。

- **準備コミットの push 前に失敗した場合**：失敗した実行の **Re-run** でバージョン準備からやり直します。`Increment and commit patch version` などで失敗し、準備コミットが `master` に入っていない場合が該当します。この状態で **Run workflow** を使うと、番号を上げずに現行バージョンを公開しようとします。
- **準備コミットが `master` に入った後に失敗した場合**：次の **Run workflow** の手順で公開を再試行します。**Re-run all jobs** では準備済みと判定され、公開をスキップしたまま成功扱いになることがあります。公開ジョブだけが失敗した場合は、**Re-run failed jobs** でも確定済みコミットの公開を再試行できます。

準備コミットが `master` に入った後の再試行手順：

1. GitHub の **Actions** で **Publish to Comfy registry** を開く。
2. **Run workflow** を選び、ブランチを `master` にする。
3. **Run workflow** で実行し、**Publish Custom Node** ステップの結果を確認する。

手動実行は現在のバージョンを公開し、番号を上げません。

## 公開設定の管理

- Registry の認証情報はリポジトリの Actions secret `REGISTRY_ACCESS_TOKEN` に保存します。
- バージョン準備ジョブは `contents: write` 権限で `master` に push します。Git の書き込み認証情報は準備ジョブの終了前に削除します。
- 公開は別の `contents: read` ジョブで行います。準備ジョブが確定したコミット SHA を checkout して公開し、composite action が参照する `github.token` も読み取り専用になります。
- 公開 Action はコミット `d2366e7abb6ab16f3bb03e3520ae25c8cf749bc9` に固定しています。Action 本体を更新するときは、更新先の内容を確認してワークフローの SHA を書き換え、コミット・PR に含めます。
- 公開 Action の `skip_checkout: 'true'` は、準備したバージョンを公開に使うために必要です。
- 公開 Action 内でインストールする `comfy-cli` は、`PIP_CONSTRAINT` で `1.22.0` に固定しています。更新時はワークフローの `Pin comfy-cli version` ステップのバージョンを変更します。
