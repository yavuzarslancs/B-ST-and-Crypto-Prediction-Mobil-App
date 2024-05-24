import React, { useState } from 'react';
import { StyleSheet, Text, View, TextInput, Button } from 'react-native';
import axios from 'axios';

const CryptoScreen = ({ navigation }) => {
  const [cryptoSymbol, setCryptoSymbol] = useState('');
  const [predictedPrice, setPredictedPrice] = useState(null);

  const handlePredict = async () => {
    console.log("Predict butonuna basıldı");
    try {
      const fullCryptoSymbol = `${cryptoSymbol}-USD`;
      const response = await axios.post('http://192.168.182.1:5000/predict', { symbol: fullCryptoSymbol, is_crypto: true });
      console.log("API'den gelen yanıt: ", response.data);
      setPredictedPrice(response.data.predicted_price);
    } catch (error) {
      console.error("Error occurred while fetching prediction: ", error);
      alert("An error occurred. Please try again.");
    }
  };

  const handleShowChart = () => {
    const fullCryptoSymbol = `${cryptoSymbol}-USD`;
    navigation.navigate('Chart', { symbol: fullCryptoSymbol, showMovingAverages: false, isCrypto: true });
  };

  const handleShowChartWithMA = () => {
    const fullCryptoSymbol = `${cryptoSymbol}-USD`;
    navigation.navigate('Chart', { symbol: fullCryptoSymbol, showMovingAverages: true, isCrypto: true });
  };

  return (
    <View style={styles.container}>
      <Text style={styles.label}>Enter Crypto Symbol:</Text>
      <TextInput
        style={styles.input}
        onChangeText={setCryptoSymbol}
        value={cryptoSymbol}
        placeholder="e.g., BTC"
        placeholderTextColor="#999"
      />
      <Button title="Predict" onPress={handlePredict} color="#007BFF" />
      {predictedPrice && (
        <Text style={styles.result}>Predicted Price: {predictedPrice}</Text>
      )}
      <View style={styles.buttonContainer}>
        <Button title="Show Crypto Chart" onPress={handleShowChart} color="#007BFF" />
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

export default CryptoScreen;
