# Comfyui-SceneDetect

PySceneDetect を使って動画からシーン境界を検出し、各シーンの代表フレームを ComfyUI の `IMAGE` バッチとして取り出すカスタムノードです。代表フレームの抽出と同時に、各シーンのメタデータを JSON (`STRING`) で返し、検出シーン数 (`INT`) も出力します。

## 特長

- 動画ファイルから自動でシーン分割（PySceneDetect 利用）
- 各シーンの代表フレームを `IMAGE` バッチで出力（開始/中間/終了を選択可）
- シーン情報を JSON で出力（フレーム番号・時刻・長さなど）
- 任意で代表フレームを JPEG として保存（サムネイル出力）

## インストール

1. 本リポジトリを ComfyUI の `custom_nodes` 配下に配置します。
   - 例: `ComfyUI/custom_nodes/Comfyui-SceneDetect`
2. 依存ライブラリをインストールします。

   ```bash
   pip install -r requirements.txt
   ```

   `requirements.txt` には以下が含まれています。
   - `scenedetect>=0.6,<0.7`
   - `opencv-python>=4.9,<5`
   - `numpy`
   - `torch`

3. ComfyUI を再起動します。

## ノード名とカテゴリ

- ノード: `PySceneDetect: Scenes → Images`
- カテゴリ: `Video/PySceneDetect`

ComfyUI 上で上記ノードが検索・配置できるようになります。

## 入出力

- 入力（required）
  - `video_path` (`STRING`): 対象動画のパス。例: `/path/to/video.mp4`
  - `method` (`content|adaptive|threshold`): シーン検出方式。
  - `threshold` (`FLOAT`): しきい値（`content/threshold` 方式で利用）。
  - `min_scene_len_sec` (`FLOAT`): 最小シーン長（秒指定）。`fps` が取得できる場合に優先。
  - `min_scene_len_frames` (`INT`): 最小シーン長（フレーム指定）。
  - `luma_only` (`BOOLEAN`): 輝度のみによる検出を行うか。

- 入力（optional）
  - `representative` (`start|middle|end`): 代表フレームの位置（開始/中間/終了）。
  - `max_width` (`INT`): 代表フレームの最大幅（0 で制限なし）。
  - `max_height` (`INT`): 代表フレームの最大高さ（0 で制限なし）。
  - `limit_scenes` (`INT`): 先頭からのシーン数制限（0 で制限なし）。
  - `write_thumbs` (`BOOLEAN`): 代表フレームを JPEG として保存するか。
  - `thumbs_dir` (`STRING`): サムネイル保存先。空なら動画と同ディレクトリに `scene_thumbs` を作成。

- 出力
  - `images` (`IMAGE`): 代表フレームのバッチ（`(B,C,H,W)`）。
  - `scenes_json` (`STRING`): シーン情報の JSON 文字列。
  - `scene_count` (`INT`): 検出シーン数。

## JSON 出力例（`scenes_json`）

```json
{
  "video_path": "/abs/path/to/video.mp4",
  "fps": 29.97,
  "method": "content",
  "threshold": 27.0,
  "min_scene_len_frames": 15,
  "representative": "start",
  "scenes": [
    {
      "index": 1,
      "start_frame": 0,
      "end_frame": 153,
      "duration_frames": 153,
      "fps": 29.97,
      "start_time": "00:00:00.000",
      "end_time": "00:00:05.105",
      "duration_sec": 5.10
    }
  ]
}
```

`scenes` 配列の各要素には、開始/終了フレーム、時間コード、フレーム長などが含まれます。

## 使い方（ComfyUI）

1. ワークスペースに `PySceneDetect: Scenes → Images` ノードを配置します。
2. `video_path` に解析したい動画ファイルを指定します。
3. 必要に応じて `method`/`threshold`/`min_scene_len_*` を調整します。
4. 代表フレームの取得位置や縮小パラメータ、サムネイル保存を設定できます。
5. 実行すると `images` 出力にシーンごとの代表フレームが、`scenes_json` にメタデータが得られます。

## ファイル構成メモ

- ルートの `__init__.py` から `nodes` を参照する ComfyUI 標準構成です。
- ノード実装は `pyscenedetect_nodes.py` および `utils/` 下のヘルパー群に分割されています。

備考: 実装は `nodes/pyscenedetect_to_images.py` にあります。共通処理は `utils/video_ops.py` に分離しています。

## トラブルシューティング

- OpenCV が動画を開けない: コーデックやパスを確認してください。`opencv-python` の代わりに `opencv-python-headless` ではないことも確認。
- PySceneDetect のバージョン不整合: `requirements.txt` の範囲で再インストールしてください。
- 出力が空（黒 1x1 のみ）: 読み込み失敗時のフォールバックです。動画やパラメータ、権限を確認してください。

## ライセンス

このリポジトリのコードは、各ファイルのライセンス表記に従います。明記がない場合はリポジトリのオーナーに従います。
