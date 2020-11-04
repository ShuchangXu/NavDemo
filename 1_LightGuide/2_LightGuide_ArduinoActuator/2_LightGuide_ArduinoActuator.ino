#include <Adafruit_NeoPixel.h>
#include <SoftwareSerial.h>

#define OFFSET 23
#define NUMPIXELS 45

#define PIN 9
#define RX 2
#define TX 3

SoftwareSerial mySerial(RX, TX);
Adafruit_NeoPixel pixels(NUMPIXELS, PIN, NEO_GRB + NEO_KHZ800);

long int receiveData = 0;

void setup() {
  Serial.begin(9600);
  mySerial.begin(9600);

  pixels.begin();
  pixels.clear();
  pixels.show();
}

void loop() {
  if (mySerial.available()) {
    if(mySerial.read() == 'y'){
      receiveData = mySerial.parseInt(); 
      if(mySerial.read() == 'x'){
        Serial.println(receiveData);
        if(receiveData == 100){
          pixels.clear();
          pixels.show();
        }
        if(receiveData >=0 && receiveData < 90){
          int id = receiveData - OFFSET;
          lightLed(id);
        }
      }
    }
  }
}

void lightLed(int id) {
  int intensity = 127;
  pixels.clear();
  if (id < 0) {
    for (int i = 0; i < -id ; i++) {
      pixels.setPixelColor(i, pixels.Color(intensity, intensity, intensity));
    }

  } else if (id > NUMPIXELS - 1) {
    for (int i = NUMPIXELS - 1; i > (2 * NUMPIXELS - 2 - id) ; i--) {
      pixels.setPixelColor(i, pixels.Color(intensity, intensity, intensity));
    }
  } else {
    pixels.setPixelColor(id, pixels.Color(intensity, intensity, intensity));
  }

  pixels.show();
}
