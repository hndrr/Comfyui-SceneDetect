# メンテナンス運用

## Comfy Registry への公開

公開設定は [publish_action.yml](.github/workflows/publish_action.yml) にあります。通常の変更は PR で `master` に集め、公開するタイミングで **Prepare release PR** を実行します。英語の要約生成・版数更新・準備 PR の作成・CI を自動で行います。要約を目視で確認して PR を `master` にマージすると、自動公開します。

リリース手順：

1. Actions の **Prepare release PR** で **Run workflow** を選び、ブランチを `master` にする。
2. 通常は入力を変更せずに実行する。`bump` は `patch`、`release_notes` は空欄でよい。必要に応じて `minor` / `major` を選ぶ。
3. 実行結果に表示された `Release v1.2.4` などの PR を開き、自動生成された英語の要約を確認する。修正するときは `.github/release-history.json` の該当版の `body` を編集し、PR 本文も揃える。
4. PR に表示される CI が成功したことを確認し、`master` にマージする。マージコミットの公開が自動で起動する。
5. Registry の版が `Active` になり、GitHub Release のタグと本文が正しいことを確認する。アップロード成功後も Registry のスキャンは別途進む。

準備 PR は自動マージしません。同じ版の準備 PR が既にある場合、再実行はその PR の URL を返し、確認中の本文は上書きしません。CI は準備ワークフロー内で PR のマージ予定コミットをテストし、その結果を PR に表示します。要約を手動で修正して準備ブランチに push した場合も、通常の **test-release-notes** が自動で実行されます。CI の手動実行承認は不要です。

要約には Copilot CLI の自動モデル選択を使います。Copilot Free の利用を有効にしたリポジトリ所有者の無料枠を使い、ワークフローから有料プランへの変更や追加利用予算の設定は行いません。生成は1回だけで、利用枠不足・アクセス不可・形式不正の場合は版数や PR を変更する前に停止します。生成内容を手動で用意する場合は `release_notes` に英語の要約を指定すると、Copilot を呼ばずに準備できます。

PR 作成直後はマージ予定コミットの生成が間に合わず、CI の checkout やコミット照合が失敗することがあります。その場合は準備ワークフローの **Re-run failed jobs** を使います。成功済みの要約生成・PR 作成ジョブはやり直さず、失敗した CI とその結果表示を再実行します。

| 変更 | `bump` | 版数の例 |
|---|---|---|
| 修正 | `patch` | `1.2.3` → `1.2.4` |
| 新機能の追加 | `minor` | `1.2.3` → `1.3.0` |
| 互換性を壊す変更 | `major` | `1.2.3` → `2.0.0` |

公開処理はキューで待機し、1件ずつ実行します。`master` で `project.version` が上がった場合だけ公開し、版数が変わらない通常の PR マージでは公開しません。公開時に追加の版数更新や `master` へのコミットは行いません。公開コミットは `master` に含まれる必要があります。

## 公開失敗時の再試行

失敗した実行の **Re-run failed jobs** を使うと、同じコミットと版数で再試行できます。GitHub Release 作成だけが失敗した場合も、この方法で再試行します。

**Run workflow** からやり直す場合は、ブランチを `master`、`mode` を `publish` にし、前の実行で checkout したコミットの40桁 SHA を `release_sha` に指定します。空欄では実行開始時の `master` を使用します。`master` に後続の変更が入っても公開内容が変わりません。既に同じ版のアップロードが成功している場合はパッケージを再公開せず、失敗した Release 作成ジョブの再実行か `sync-notes` を使います。

## 公開設定の管理

- 配布 ZIP では `.comfyignore` で `.github/` と `tests/` を除外します。GitHub Actions とテストはリポジトリ上で引き続き利用できます。
- Registry の認証情報はリポジトリの Actions secret `REGISTRY_ACCESS_TOKEN` に保存します。
- リリース準備 PR の自動作成には、Settings → Actions → General の **Allow GitHub Actions to create and approve pull requests** を有効にします。既定のトークン権限は読み取り専用のまま、準備 PR 作成ジョブに `contents: write` と `pull-requests: write`、CI 結果の表示ジョブに `checks: write` を付与します。
- 要約生成ジョブは `contents: read` と `copilot-requests: write` で動き、標準の `GITHUB_TOKEN` で認証します。追加の PAT は不要です。前回のリリースタグからの変更差分を渡し、テストと既存リリース本文は入力から除きます。Copilot のツール利用を無効にし、リポジトリ内の追加指示ファイルは読み込みません。入力が大きすぎる場合は途中で切り捨てず停止します。
- 公開対象の確認・Registry 公開ジョブは `contents: read` で動き、Git の認証情報は checkout 後に保持しません。
- 公開は別の `contents: read` ジョブで行います。準備ジョブが確定したコミット SHA を checkout して公開し、composite action が参照する `github.token` も読み取り専用になります。
- 公開 Action はコミット `d2366e7abb6ab16f3bb03e3520ae25c8cf749bc9` に固定しています。Action 本体を更新するときは、更新先の内容を確認してワークフローの SHA を書き換え、コミット・PR に含めます。
- 公開 Action の `skip_checkout: 'true'` は、準備したバージョンを公開に使うために必要です。
- 公開 Action 内でインストールする `comfy-cli` は、`PIP_CONSTRAINT` で `1.22.0` に固定しています。更新時はワークフローの `Pin comfy-cli version` ステップのバージョンを変更します。

## GitHub Releases と更新内容

Registry 公開に成功した後、公開したコミットを指す `v1.2.3` などのタグと GitHub Release を作ります。更新内容は英語で統一し、利用者に影響する機能追加・修正を1〜3項目にまとめます。公開手順や削除済み版の経緯など、運用上の説明は含めません。PR タイトルの列挙や作者・PR リンクは本文に含めず、リリース全体の変更内容を説明します。

自動生成または **Prepare release PR** に入力した要約は、公開する版の `.github/release-history.json` に保存します。例：現在が `1.2.3` で `patch` を選ぶと、`"1.2.4": {"body": "- Fix ..."}` を追加します。要約は PR 内で編集できます。公開する本文はこの記録ファイルの内容なので、PR 本文だけを修正しても公開本文は変わりません。マージ後の SHA はまだ不明なので省略できます。本文がない場合は公開前に停止します。手動の `publish` 実行では `release_notes` に本文を指定でき、その入力を記録ファイルより優先します。

固定済み `comfy-cli` の `COMFY_NODE_CHANGELOG` と GitHub Release に、同じ要約を渡します。既存 Release の本文を編集した場合は、後続の同期でその本文を使用します。

Release 作成は別の `contents: write` ジョブで行います。Registry 公開ジョブの `github.token` は `contents: read` のままです。同じタグが別のコミットを指す場合や既存 Release が下書きの場合は停止し、既存のタグ・本文を上書きしません。

GitHub Release 作成だけが失敗した場合は **Re-run failed jobs** で再試行できます。既に Registry 公開が成功している版の履歴だけを補完するときは、次の `sync-notes` を使います。

## 公開済み版の履歴補完・本文の同期

既存の Registry バージョンに GitHub Release がない場合や、公開済みの本文を揃える場合は `sync-notes` を使います。

1. Actions の **Publish to Comfy registry** で **Run workflow** を選ぶ。
2. ブランチを `master`、`mode` を `sync-notes` にして実行する。
3. **Sync notes for existing Registry versions** ジョブの結果を確認する。

このモードは `master` でのみ実行できます。既存の Registry バージョンだけを対象にし、版数を上げたりパッケージを再公開したりしません。GitHub Release がなければ作り、同じ本文を Registry の `changelog` に保存します。既存版の `deprecated` 状態は維持します。GitHub Release の本文を編集してから再度実行すると、その内容を Registry に反映できます。

公開済みのタグも記録ファイルの SHA もない新しい版では、まず失敗した Release 作成ジョブを再実行します。元の実行を利用できない場合は、公開したコミットの SHA を記録ファイルに追加してから `sync-notes` を実行します。

公開済み版の文言を変更するときは、先に `.github/release-history.json` の変更を PR で確認し、マージ後に GitHub Release の本文を同じ文言に更新してから `sync-notes` を実行します。既存の GitHub Release がある場合、記録ファイルの変更だけでは公開済みの本文は上書きされません。

過去版のコミットと更新内容は `.github/release-history.json` に記録しています。`1.0.0`、`1.0.1`、`1.1.0`、`1.2.0`、`1.2.2` のコミットは、実際の Registry 公開パッケージと照合済みです。削除済みの `1.2.1` は作らず、変更内容を `1.2.2` にまとめています。SHA を省略した版の履歴補完では、旧運用の `Prepare registry version ...` コミットか、公開済みの `v<版数>` タグを使用し、版数・タグ・`master` の履歴を検証します。
