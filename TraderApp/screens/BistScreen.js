import React, { useState } from 'react';
import { StyleSheet, Text, View, TextInput, Button } from 'react-native';
import axios from 'axios';

const BistScreen = ({ navigation }) => {
  const [stockSymbol, setStockSymbol] = useState('');
  const [predictedPrice, setPredictedPrice] = useState(null);
  const [predictionDate, setPredictionDate] = useState(null);

  const handlePredict = async () => {
    console.log("Predict butonuna basıldı");
    try {
      const fullStockSymbol = `${stockSymbol}.IS`;
      const response = await axios.post('http://"Your IP KEY"/predict', { symbol: fullStockSymbol, is_crypto: false });
      console.log("API'den gelen yanıt: ", response.data);
      setPredictedPrice(response.data.predicted_price);
      setPredictionDate(response.data.date);
    } catch (error) {
      console.error("Error occurred while fetching prediction: ", error);
      alert("An error occurred. Please try again.");
    }
  };

  const handleShowChart = () => {
    const fullStockSymbol = `${stockSymbol}.IS`;
    navigation.navigate('ChartBist', { symbol: fullStockSymbol, showMovingAverages: false, isCrypto: false });
  };

  const handleShowChartWithMA = () => {
    const fullStockSymbol = `${stockSymbol}.IS`;
    navigation.navigate('ChartBist', { symbol: fullStockSymbol, showMovingAverages: true, isCrypto: false });
  };

  return (
    <View style={styles.container}>
      <Text style={styles.label}>Enter BIST Stock Symbol:</Text>
      <TextInput
        style={styles.input}
        onChangeText={setStockSymbol}
        value={stockSymbol}
        placeholder="e.g., XU100"
        placeholderTextColor="#999"
      />
      <Button title="Predict" onPress={handlePredict} color="#007BFF" />
      {predictedPrice && (
        <Text style={styles.result}>Predicted Price for {predictionDate}: {predictedPrice}</Text>
      )}
      <View style={styles.buttonContainer}>
        <Button title="Show Stock Chart" onPress={handleShowChart} color="#007BFF" />
      </View>
      <View style={styles.buttonContainer}>
        <Button title="Show Chart with Moving Averages" onPress={handleShowChartWithMA} color="#007BFF" />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#f8f8f8',
  },
  label: {
    fontSize: 18,
    color: '#333',
    marginBottom: 8,
  },
  input: {
    height: 40,
    borderColor: 'gray',
    borderWidth: 1,
    width: '100%',
    marginBottom: 12,
    paddingLeft: 8,
    backgroundColor: '#fff',
    color: '#333',
  },
  result: {
    marginTop: 20,
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  buttonContainer: {
    marginTop: 12,
    width: '100%',
  },
});

export default BistScreen;
