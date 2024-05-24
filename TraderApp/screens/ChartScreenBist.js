import React, { useEffect, useState } from 'react';
import { StyleSheet, Text, View, Dimensions } from 'react-native';
import Plotly from 'react-native-plotly';
import axios from 'axios';

const screenWidth = Dimensions.get('window').width;

const ChartBist = ({ route }) => {
  const { symbol, showMovingAverages, isCrypto } = route.params;
  const [chartData, setChartData] = useState(null);

  useEffect(() => {
    if (showMovingAverages) {
      fetchLast200DaysChartData();
    } else {
      fetchFullChartData();
    }
  }, [symbol]);

  const fetchFullChartData = async () => {
    try {
      const response = await axios.post('http://192.168.182.1:5000/plot_chart_interactive', { crypto_symbol: symbol });
      const data = response.data;
      const formattedData = [
        { x: data.dates, y: data.prices, type: 'scatter', mode: 'lines', name: 'Kapanış Fiyatları' }
      ];
      setChartData(formattedData);
    } catch (error) {
      console.error("Error occurred while fetching full chart data: ", error);
      alert("An error occurred while fetching full chart data. Please check the symbol and try again.");
    }
  };

  const fetchLast200DaysChartData = async () => {
    try {
      const response = await axios.post('http://192.168.182.1:5000/plot_moving_averages', { crypto_symbol: symbol });
      const data = response.data;
      const today = new Date();
      const startDate = new Date(today);
      startDate.setDate(today.getDate() - 200);

      const filteredData = data.dates.reduce((acc, date, index) => {
        const currentDate = new Date(date);
        if (currentDate >= startDate && currentDate <= today) {
          acc.dates.push(date);
          acc.prices.push(data.prices[index]);
          if (data.ma20) acc.ma20.push(data.ma20[index]);
          if (data.ma50) acc.ma50.push(data.ma50[index]);
          if (data.ma100) acc.ma100.push(data.ma100[index]);
          if (data.ma200) acc.ma200.push(data.ma200[index]);
        }
        return acc;
      }, { dates: [], prices: [], ma20: [], ma50: [], ma100: [], ma200: [] });

      let formattedData = [
        { x: filteredData.dates, y: filteredData.prices, type: 'scatter', mode: 'lines', name: 'Kapanış Fiyatları' }
      ];

      if (showMovingAverages) {
        formattedData.push(
          { x: filteredData.dates, y: filteredData.ma20, type: 'scatter', mode: 'lines', name: '20 Günlük Ortalama' },
          { x: filteredData.dates, y: filteredData.ma50, type: 'scatter', mode: 'lines', name: '50 Günlük Ortalama' },
          { x: filteredData.dates, y: filteredData.ma100, type: 'scatter', mode: 'lines', name: '100 Günlük Ortalama' },
          { x: filteredData.dates, y: filteredData.ma200, type: 'scatter', mode: 'lines', name: '200 Günlük Ortalama' }
        );
      }

      setChartData(formattedData);
    } catch (error) {
      console.error("Error occurred while fetching last 200 days chart data: ", error);
      alert("An error occurred while fetching last 200 days chart data. Please check the symbol and try again.");
    }
  };

  const layout = {
    title: showMovingAverages ? `Stock Chart for ${symbol} - Last 200 Days` : `Stock Chart for ${symbol} - All Data`,
    paper_bgcolor: 'white',
    plot_bgcolor: 'white',
    font: {
      color: 'black'
    },
    xaxis: {
      title: 'Tarih',
      showgrid: true,
      zeroline: true,
      color: 'black'
    },
    yaxis: {
      title: 'Fiyat (USD)',
      showgrid: true,
      zeroline: true,
      color: 'black'
    },
    xaxis_rangeslider_visible: true,
    dragmode: 'pan',
  };

  const config = {
    scrollZoom: true,  // Mouse scroll ile zoom yapma özelliğini etkinleştirir
    responsive: true
  };

  return (
    <View style={styles.container}>
      {chartData ? (
        <Plotly
          data={chartData}
          layout={layout}
          config={config}
          style={styles.plotly}
        />
      ) : (
        <Text style={styles.loadingText}>Loading...</Text>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingTop: 40,
    backgroundColor: 'white',
  },
  plotly: {
    width: screenWidth,
    height: '100%',
  },
  loadingText: {
    color: 'black',
    fontSize: 18,
    textAlign: 'center',
    marginTop: 20,
  },
});

export default ChartBist;
