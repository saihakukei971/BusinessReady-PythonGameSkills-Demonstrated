プロジェクト概要
1. テトリス (tetris.py)
Pygameを使った古典的なブロック落下パズルゲームです。キーボード操作で、様々な形のブロックを回転させながら落とし、横一列を揃えて消していきます。ゲームオーバーまでどれだけ高得点を取れるかを競います。
コードでは、ブロックの衝突判定、落下速度の調整、行の完成判定と削除などの複雑なロジックを実装しています。これらは在庫管理システムの空間最適化や条件判定ロジックと同様の複雑さを持ちます。
2. ぷよぷよ (puyopuyo.py)
カラフルな「ぷよ」を操作して落とし、同じ色のぷよが4つ以上つながると消える人気パズルゲームの再現です。消えた後に残ったぷよが落ちて新たな連鎖が発生するシステムが特徴です。
このゲームでは、連鎖判定アルゴリズム（再帰的な連結成分検出）、物理的な落下処理（アニメーション付き）、複雑な状態管理を実装しています。これらのスキルは、サプライチェーン影響分析や、複雑なビジネスプロセスフローの実装と直接関連しています。
3. 強化学習ホッケー (hyper_hockey.py)
2つのAIエージェントが対戦するエアホッケーシミュレーションです。強化学習（Proximal Policy Optimization）を使って両エージェントを訓練し、互いに競争しながら戦略を学習します。
このプロジェクトでは、物理シミュレーション、予測アルゴリズム、多エージェント強化学習などの高度な技術を実装しています。これらは、市場予測モデル、競争市場シミュレーション、ビジネスインテリジェンスなどの複雑なビジネスアプリケーションに直接応用可能です。
4. RPG ゲーム (main.py)
Pygameで作成したシンプルなRPGゲームです。プレイヤーキャラクターをマップ上で操作し、敵との戦闘や別マップへの移動など、RPGゲームの基本機能を実装しています。
このゲームでは、マップ描画、キャラクター制御、敵AIの自動移動、衝突判定、戦闘システム、セーブ＆ロード機能など、多くの機能を統合しています。これらは、リアルタイムユーザーインタラクション処理、永続的なデータ管理、状態遷移など、業務システムでも重要なコンセプトを実践的に示しています。
リポジトリ構成
python-game-portfolio/
├── tetris/
│   ├── tetris.py          # テトリスゲーム実装
│   ├── requirements.txt   # 必要なライブラリ
│   └── assets/            # 画像リソース
├── puyopuyo/
│   ├── puyopuyo.py        # ぷよぷよゲーム実装
│   ├── requirements.txt   # 必要なライブラリ
│   └── assets/            # 画像リソース
├── rl-hockey/
│   ├── hyper_hockey.py    # 強化学習AIホッケー実装
│   ├── requirements.txt   # 必要なライブラリ
│   └── models/            # 学習済みモデル
├── rpg/
│   ├── main.py            # RPGゲームのメインスクリプト
│   ├── map1.py            # 最初のマップデータ 
│   ├── map2.py            # 二つ目のマップデータ
│   ├── assets/            # 画像リソース
│   │   ├── character.png  # プレイヤーキャラクター画像
│   │   ├── enemy_1.png    # 敵キャラクター画像
│   │   ├── path.png       # 道の画像
│   │   ├── rock.png       # 岩の画像
│   │   └── to_next_map.png # マップ移動ポイント画像
│   └── requirements.txt   # 必要なライブラリ
├── common/
│   ├── __init__.py
│   └── utils.py           # 共通ユーティリティ関数
├── requirements.txt       # プロジェクト全体の依存関係
└── README.md              # このファイル
インストールと実行方法
bash# リポジトリのクローン
git clone https://github.com/yourusername/python-game-portfolio.git
cd python-game-portfolio

# 仮想環境のセットアップ
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate

# 全ての依存関係のインストール
pip install -r requirements.txt

# 特定のゲームを実行する場合
# テトリス
cd tetris
python tetris.py

# ぷよぷよ
cd ../puyopuyo
python puyopuyo.py

# 強化学習ホッケー
cd ../rl-hockey
python hyper_hockey.py

# RPGゲーム
cd ../rpg
python main.py
RPGゲームの特徴
RPGゲームでは、以下の機能を実装しています：

マップ描画・スクロール：10×10タイルから40×40タイルのマップを実装し、キャラクターの移動に合わせてマップがスクロール
キャラクター操作：矢印キーで移動できるプレイヤーキャラクター
敵キャラクターAI：ランダムな方向に自動で移動する敵キャラクター
衝突判定：プレイヤーと敵の衝突判定、地形の当たり判定
バトルシステム：敵と接触した際のタイミングバトル（バーの動きに合わせてキーを押すリズムゲーム形式）
セーブ＆ロード機能：JSONファイルを使用したゲーム状態の保存と読み込み
ゲームオーバー処理：プレイヤーが敗北した際のデータ初期化
マップ切り替え：特定のポイントで別マップへの移動

ビジネスアプリケーションへの応用
これらのゲーム開発で培った技術は、以下のようなビジネスシステムに直接応用できます：

リアルタイムデータ処理: 60FPSでの描画処理はリアルタイムダッシュボードやモニタリングシステムと同様の要件
複雑なビジネスルール実装: 消去判定や連鎖処理は、複雑な取引ルールやコンプライアンス条件と同等の複雑さ
予測分析と最適化: AIホッケーの予測アルゴリズムは、在庫予測や需要予測などのビジネスインテリジェンスに応用可能
エラー耐性と復旧: ゲームが途切れずに動くための耐障害設計は、ミッションクリティカルな業務システムに必須
データ永続化: RPGのセーブ機能に見られるデータの永続化は、トランザクション処理とデータ整合性の実装例

このポートフォリオは、複雑なビジネスロジックとシステム設計の実装能力を証明するために、あえてゲーム開発という挑戦的な分野で作成しました。
Gitにアップロードする方法
1. リポジトリの初期化
bash# プロジェクトフォルダに移動
cd ~/path/to/python-game-portfolio

# Gitリポジトリの初期化
git init

# .gitignoreファイルの作成
echo "venv/" > .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo "*.pyo" >> .gitignore
echo "*.pyd" >> .gitignore
echo ".Python" >> .gitignore
echo "env/" >> .gitignore
echo "build/" >> .gitignore
echo "develop-eggs/" >> .gitignore
echo "dist/" >> .gitignore
echo "downloads/" >> .gitignore
echo "eggs/" >> .gitignore
echo ".eggs/" >> .gitignore
echo "lib/" >> .gitignore
echo "lib64/" >> .gitignore
echo "parts/" >> .gitignore
echo "sdist/" >> .gitignore
echo "var/" >> .gitignore
echo "*.egg-info/" >> .gitignore
echo "*.egg" >> .gitignore
echo ".idea/" >> .gitignore
echo ".DS_Store" >> .gitignore
echo "save_data.json" >> .gitignore  # RPGのセーブデータ
2. 必要なファイルの作成とコミット
bash# READMEとrequirements.txtの作成（内容は上記に基づいて作成）

# ディレクトリ構造の作成
mkdir -p tetris/assets puyopuyo/assets rl-hockey/models rpg/assets common

# 上記のREADME内容をREADME.mdファイルに保存

# requirements.txtの作成（全体の依存関係）
echo "pygame==2.1.2" > requirements.txt
echo "numpy==1.23.5" >> requirements.txt
echo "torch==2.0.0" >> requirements.txt
echo "torchvision==0.15.0" >> requirements.txt
echo "matplotlib==3.7.1" >> requirements.txt
echo "imageio==2.25.0" >> requirements.txt

# 各ゲームディレクトリにそれぞれのrequirements.txtを作成
echo "pygame==2.1.2" > tetris/requirements.txt
echo "pygame==2.1.2" > puyopuyo/requirements.txt
echo "pygame==2.1.2
numpy==1.23.5
torch==2.0.0
torchvision==0.15.0
matplotlib==3.7.1
imageio==2.25.0" > rl-hockey/requirements.txt
echo "pygame==2.1.2" > rpg/requirements.txt

# 共通モジュールの初期化
touch common/__init__.py
3. ソースコードファイルを各ディレクトリに配置

tetris.pyをtetris/ディレクトリに配置
puyopuyo.pyをpuyopuyo/ディレクトリに配置
hyper_hockey.pyをrl-hockey/ディレクトリに配置
RPGゲーム関連ファイル(main.py, map1.py, map2.py)をrpg/ディレクトリに配置
各画像ファイルを対応するassets/ディレクトリに配置

4. Gitへのコミットとプッシュ
bash# 全てのファイルをステージング
git add .

# 初期コミット
git commit -m "Initial commit: Python Game Portfolio"

# GitHubにリポジトリを作成し、リモートリポジトリを追加
git remote add origin https://github.com/yourusername/python-game-portfolio.git

# プッシュ
git push -u origin master
5. 注意点

画像リソースは対応するゲームのassets/ディレクトリに配置し、コード内のパスを適切に修正する必要があります。
ゲームコード内のファイルパス参照を相対パスに修正します：
python# 例：RPGゲームの画像読み込み部分
path_img = pygame.image.load("assets/path.png")
rock_img = pygame.image.load("assets/rock.png")
character_img = pygame.image.load("assets/character.png")
enemy_1_img = pygame.image.load("assets/enemy_1.png")
to_next_map_img = pygame.image.load("assets/to_next_map.png")

各ゲームディレクトリに簡単な実行手順を記載したREADME.mdを追加するとより親切です。
