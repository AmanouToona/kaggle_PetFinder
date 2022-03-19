# lata submission memo

## ToDo

- [batch norm to eval mode](https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/301259) が精度をあげるのか確認すること。
- [モデルの一部を固定すること](https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/301015)を試す。 @cdeotte what's the reason behind freezing 66% of the backbone and training the rest + the head? Could i also know how you can unfreeze that 33% of the backbone for train? Thanks and congrats on 6th!
- [fastai notebook](https://www.kaggle.com/tanlikesmath/petfinder-pawpularity-eda-fastai-starter)を参考にもっともよいモデルを探すこと。
- 入力する画像サイズを増やすこと。
- 大きなモデルでは、augmentation を増やす。 Random square crop, Horizontal Flip, Vertical Flip, Transpose, Brightness, Contrast, Hue, Saturation, Rotation, Random Erase.
- [モデルの多様性を増やす方法](https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/301015#:~:text=1-,They%20need%20to%20be%20different,-can%20you%20explain)Here are some ideas that come to mind

## 試行結果のメモ

### 1

initialize の効き方の調査
config: latesub_001, latesub_002
xavier の初期化を入れると精度が向上するか？
手元の CV では精度の向上が見られた。
微妙な差。

## 2

batch norm to eval mode
手元ではむしろ精度が悪化する。 バグってるかも。
config: latesub_006, latesub_007
batch size に依存する？
config: latesub_008, latesub_009
単純に accumulation を切っただけでは変化なし。 (batch 変えていないので、当然と言えば当然)

そもそも、他の層に影響のあるプログラムになっていないか？ swim でためす。
config 010: 001 に比べて精度が悪い。 initialize の影響？, config001 -> config 011: normalization 内包が精度がよいようだ。
012: = 001 やはり悪い
013: = 012 から normalize 削除 やや良くなる
014: = 013 から bn_eval を false に変更
014, 015: model efficientnet02, bn_freeze あり、なし、で微妙に固めた方がよさそう。
016, 017: model resnet50
