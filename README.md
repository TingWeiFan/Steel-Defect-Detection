# Steel-Defect-Detection

此專案是Kaggle上的競賽，您可以從下方連結得到更詳細的比賽資訊  
https://www.kaggle.com/c/severstal-steel-defect-detection/overview

參考網路上的範例，並根據本身的裝置環境修改了資料及數量、模型參數與模型架構

## 資料集
從Kaggle下載的資料集總共有四個類別，從下圖可以知道類別之間的資料量不平均，且某些資料量明顯偏少。  
<img src="img/label_number.jpeg" width="40%"/>

為了解決資料不平均的問題，我先做資料增強(data augmentation)，再將每種類別取差不多數量的數據做訓練，但準確率卻沒有明顯改善，最後做法為直接將資料增強後的數據全部作為training data。  
執行下面兩個python檔後，會在名為data的資料夾下產生train, validation, test用的csv檔  
```
python data_augmentation.py
python split_train_test_csv.py
```

## 模型

## 評估

## 結果
