import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { StyleSheet, Text, View, TouchableOpacity } from 'react-native';
import CryptoScreen from './screens/CryptoScreen';
import BistScreen from './screens/BistScreen';
import ChartScreen from './screens/ChartScreen';
import ChartScreenBist from './screens/ChartScreenBist';

const Stack = createStackNavigator();

function HomeScreen({ navigation }) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>TraderApp</Text>
      <TouchableOpacity
        style={styles.button}
        onPress={() => navigation.navigate('BIST')}>
        <Text style={styles.buttonText}>BIST</Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={styles.button}
        onPress={() => navigation.navigate('Crypto')}>
        <Text style={styles.buttonText}>Crypto</Text>
      </TouchableOpacity>
    </View>
  );
}

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Home">
        <Stack.Screen name="Home" component={HomeScreen} options={{ headerShown: false }} />
        <Stack.Screen name="BIST" component={BistScreen} />
        <Stack.Screen name="Crypto" component={CryptoScreen} />
        <Stack.Screen name="Chart" component={ChartScreen} />
        <Stack.Screen name="ChartBist" component={ChartScreenBist} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'red',
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    fontSize: 32,
    color: 'black',
    marginBottom: 40,
  },
  button: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 5,
    marginBottom: 20,
    width: '80%',
    alignItems: 'center',
  },
  buttonText: {
    fontSize: 18,
    color: 'black',
  },
});
