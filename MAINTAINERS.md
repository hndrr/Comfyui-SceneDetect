# メンテナンス運用

## Comfy Registry への公開

公開設定は [publish_action.yml](.github/workflows/publish_action.yml) にあります。`master` への push・PR マージで自動公開が起動します。変更が `.github/**`、`README.md`、`MAINTAINERS.md` だけの場合は、自動公開もバージョン更新も起動しません。

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
2. **Run workflow** を選び、ブランチを `master`、`mode` を `publish` にする。
3. **Run workflow** で実行し、**Publish Custom Node** ステップの結果を確認する。

手動実行は現在のバージョンを公開し、番号を上げません。

## 公開設定の管理

- 配布 ZIP では `.comfyignore` で `.github/` と `tests/` を除外します。GitHub Actions とテストはリポジトリ上で引き続き利用できます。
- Registry の認証情報はリポジトリの Actions secret `REGISTRY_ACCESS_TOKEN` に保存します。
- バージョン準備ジョブは `contents: write` 権限で `master` に push します。Git の書き込み認証情報は準備ジョブの終了前に削除します。
- 公開は別の `contents: read` ジョブで行います。準備ジョブが確定したコミット SHA を checkout して公開し、composite action が参照する `github.token` も読み取り専用になります。
- 公開 Action はコミット `d2366e7abb6ab16f3bb03e3520ae25c8cf749bc9` に固定しています。Action 本体を更新するときは、更新先の内容を確認してワークフローの SHA を書き換え、コミット・PR に含めます。
- 公開 Action の `skip_checkout: 'true'` は、準備したバージョンを公開に使うために必要です。
- 公開 Action 内でインストールする `comfy-cli` は、`PIP_CONSTRAINT` で `1.22.0` に固定しています。更新時はワークフローの `Pin comfy-cli version` ステップのバージョンを変更します。

## GitHub Releases と更新内容

Registry 公開に成功した後、公開したコミットを指す `v1.2.3` などのタグと GitHub Release を作ります。更新内容は英語で統一し、利用者に影響する機能追加・修正を1〜3項目にまとめます。公開手順や削除済み版の経緯など、運用上の説明は含めません。自動生成に使う PR タイトルも英語にします。更新内容は公開前に GitHub のマージ済み PR から生成し、固定済み `comfy-cli` の `COMFY_NODE_CHANGELOG` と GitHub Release の本文に同じ内容を渡します。手動の `publish` 実行では `release_notes` に本文を指定することもできます。

Release 作成は別の `contents: write` ジョブで行います。Registry 公開ジョブの `github.token` は `contents: read` のままです。同じタグが別のコミットを指す場合や既存 Release が下書きの場合は停止し、既存のタグ・本文を上書きしません。

GitHub Release 作成だけが失敗した場合は **Re-run failed jobs** で再試行できます。既に Registry 公開が成功している版の履歴だけを補完するときは、次の `sync-notes` を使います。

## 公開済み版の履歴補完・本文の同期

導入時に過去の GitHub Releases がまだない場合は、次の通常公開より先に `sync-notes` を実行して履歴を補完します。これにより、次の版の自動生成本文に過去の変更全体が含まれるのを防ぎます。

1. Actions の **Publish to Comfy registry** で **Run workflow** を選ぶ。
2. ブランチを `master`、`mode` を `sync-notes` にして実行する。
3. **Sync notes for existing Registry versions** ジョブの結果を確認する。

このモードは `master` でのみ実行できます。既存の Registry バージョンだけを対象にし、版数を上げたりパッケージを再公開したりしません。GitHub Release がなければ作り、同じ本文を Registry の `changelog` に保存します。既存版の `deprecated` 状態は維持します。GitHub Release の本文を編集してから再度実行すると、その内容を Registry に反映できます。

公開済み版の文言を変更するときは、先に `.github/release-history.json` の変更を PR で確認し、マージ後に GitHub Release の本文を同じ文言に更新してから `sync-notes` を実行します。既存の GitHub Release がある場合、記録ファイルの変更だけでは公開済みの本文は上書きされません。

過去版のコミットと更新内容は `.github/release-history.json` に記録しています。`1.0.0`、`1.0.1`、`1.1.0`、`1.2.0`、`1.2.2` のコミットは、実際の Registry 公開パッケージと照合済みです。削除済みの `1.2.1` は作らず、変更内容を `1.2.2` にまとめています。
