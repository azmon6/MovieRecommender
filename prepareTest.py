import pandas as pd

#Tuk prosto vzimam obshtite filmi mejdu dvata file-a koito imame
#Hardcode-nati sa koi featuri izpolzvame

csv_file = 'movies_test.csv'
testDF = pd.read_csv(csv_file, sep=';')

csv_file2 = 'smallDataset.csv'
trainDF = pd.read_csv(csv_file2)

result = pd.merge(testDF, trainDF[['Series_Title']], on='Series_Title', how='inner')

result['IMDB_Rating'] /= 10

result['Released_Year'] = pd.to_datetime(result['Released_Year'])
result['Released_Year'] = result['Released_Year'].dt.year

print(result['IMDB_Rating'])

result.to_csv("testDataset.csv", sep=',',index=False)