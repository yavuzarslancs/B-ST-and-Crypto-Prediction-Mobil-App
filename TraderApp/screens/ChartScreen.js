import React, { useEffect, useState } from 'react';
import { StyleSheet, Text, View, Dimensions } from 'react-native';
import Plotly from 'react-native-plotly';
import axios from 'axios';

const screenWidth = Dimensions.get('window').width;

const ChartScreen = ({ route }) => {
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
      const response = await axios.get(`https://query1.finance.yahoo.com/v8/finance/chart/${symbol}`, {
        params: {
          interval: '1d',
          range: 'max' // Tüm verileri çekiyoruz
        }
      });

      if (response.data.chart.result) {
        const data = response.data.chart.result[0];
        const timestamps = data.timestamp.map(ts => new Date(ts * 1000).toLocaleDateString());
        const closePrices = data.indicators.quote[0].close;

        const formattedData = [
          { x: timestamps, y: closePrices, type: 'scatter', mode: 'lines', name: 'Kapanış Fiyatları' }
        ];

        setChartData(formattedData);
      } else {
        throw new Error('Invalid symbol or no data found');
      }
    } catch (error) {
      console.error("Error occurred while fetching full chart data: ", error);
      alert("An error occurred while fetching full chart data. Please check the symbol and try again.");
    }
  };

  const fetchLast200DaysChartData = async () => {
    try {
      const response = await axios.get(`https://query1.finance.yahoo.com/v8/finance/chart/${symbol}`, {
        params: {
          interval: '1d',
          range: '1y' // Son 1 yılı çekiyoruz
        }
      });

      if (response.data.chart.result) {
        const data = response.data.chart.result[0];
        const timestamps = data.timestamp.map(ts => new Date(ts * 1000));
        const closePrices = data.indicators.quote[0].close;

        // 200 gün öncesinden bugüne kadar olan verileri filtreleyelim
        const today = new Date();
        const startDate = new Date(today);
        startDate.setDate(today.getDate() - 200);

        const filteredData = timestamps.reduce((acc, date, index) => {
          if (date >= startDate && date <= today) {
            acc.timestamps.push(date.toLocaleDateString());
            acc.closePrices.push(closePrices[index]);
          }
          return acc;
        }, { timestamps: [], closePrices: [] });

        let formattedData = [
          { x: filteredData.timestamps, y: filteredData.closePrices, type: 'scatter', mode: 'lines', name: 'Kapanış Fiyatları' }
        ];

        if (showMovingAverages) {
          // Hesaplanan hareketli ortalamalar
          const ma20 = calculateMovingAverage(filteredData.closePrices, 20);
          const ma50 = calculateMovingAverage(filteredData.closePrices, 50);
          const ma100 = calculateMovingAverage(filteredData.closePrices, 100);
          const ma200 = calculateMovingAverage(filteredData.closePrices, 200);

          formattedData = [
            { x: filteredData.timestamps, y: filteredData.closePrices, type: 'scatter', mode: 'lines', name: 'Kapanış Fiyatları' },
            { x: filteredData.timestamps, y: ma20, type: 'scatter', mode: 'lines', name: '20 Günlük Ortalama' },
            { x: filteredData.timestamps, y: ma50, type: 'scatter', mode: 'lines', name: '50 Günlük Ortalama' },
            { x: filteredData.timestamps, y: ma100, type: 'scatter', mode: 'lines', name: '100 Günlük Ortalama' },
            { x: filteredData.timestamps, y: ma200, type: 'scatter', mode: 'lines', name: '200 Günlük Ortalama' },
          ];
        }

        setChartData(formattedData);
      } else {
        throw new Error('Invalid symbol or no data found');
      }
    } catch (error) {
      console.error("Error occurred while fetching last 200 days chart data: ", error);
      alert("An error occurred while fetching last 200 days chart data. Please check the symbol and try again.");
    }
  };

  const calculateMovingAverage = (data, windowSize) => {
    let result = [];
    for (let i = 0; i < data.length; i++) {
      if (i < windowSize) {
        result.push(null);
      } else {
        const windowData = data.slice(i - windowSize, i);
        const sum = windowData.reduce((acc, curr) => acc + curr, 0);
        result.push(sum / windowSize);
      }
    }
    return result;
  };

  const layout = {
    title: showMovingAverages ? `Crypto Chart for ${symbol} - Last 200 Days` : `Crypto Chart for ${symbol} - All Data`,
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

export default ChartScreen;
