#include <Ethernet.h>
#include <PubSubClient.h>
#include <SPI.h>
#include <MFRC522.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

int SS_PIN = 8;
int RST_PIN = 9;
int SS_Ethernet = 10;

unsigned long firsttime = 0;
unsigned long secondtime = 0;
long mil=0, mil_new=0;

const int role = 7;
int role_durum = 0;
MFRC522 rfid(SS_PIN, RST_PIN);
MFRC522::MIFARE_Key key;

byte nuidPICC[4];

String ID;

byte mac[] = {0xD8, 0x57, 0xEF, 0x66, 0x33, 0x74};
IPAddress ip(192, 168, 1, 100);
EthernetServer server(1883);//use port 80 - the standard for HTTP
//***************************************************************
EthernetClient ethclient;
PubSubClient client(ethclient);

const char* ssid = "<wifi_ssid>";
const char* password = "<wifi_pass>";
const char* mqtt_server = "192.168.1.2";
const char* mqtt_user = "admin";
const char* mqtt_pass = "123456";

long lastTime, now;
String station = "123";
String Time = "15.22.35";
String port = "6";
//String msg = "{Station:" + station + "," + "ID:" + ID + "," + "Time:" + Time + "," + "Port:" + port + "}";
String msg;
char payload[50] = "msg";

int turk = 0, ingiliz = 1, alman = 2;
LiquidCrystal_I2C lcd(0x27, 16, 2);

char hosgeldiniz[] = "HOSGELDINIZ";
int size_hosgeldiniz = sizeof(hosgeldiniz);
String iyi_eglenceler[] = {"IYI EGLENCELER", "HAVE FUN", "HABE SPA", "HOLA", "EHLEN"};
boolean kontrol = true;
String messageTemp;

byte gulen_yuz[8] = {
  0b00100,
  0b00010,
  0b01001,
  0b00001,
  0b00001,
  0b01001,
  0b00010,
  0b00100
};
byte kalp[8] = {
  0b00000,
  0b01010,
  0b11111,
  0b11111,
  0b01110,
  0b00100,
  0b00000,
  0b00000
};
byte ss[8] = {
  0b01110,
  0b10001,
  0b10010,
  0b10100,
  0b10010,
  0b10001,
  0b10010,
  0b10100
};

void hosgeldiniz_print() {
  lcd.clear();
  lcd.setCursor(0, 0);
  String msg;
  for (int i = 1; i < size_hosgeldiniz; ++i) {
    msg += (char)hosgeldiniz[i - 1];
    lcd.setCursor(2, 0);
    lcd.print(msg);
    delay(100);
  }
  lcd.write((byte)0);
  lcd.setCursor(1, 0);
  lcd.print("-");
  lcd.setCursor(14, 0);
  lcd.print("-");
  delay(100);
  lcd.setCursor(0, 0);
  lcd.print("-");
  lcd.setCursor(15, 0);
  lcd.print("-");
}

void kart_donut(int l) {
  int size_donut = (16 - iyi_eglenceler[l].length()) / 2;
  //Serial.println(size_donut);
  int cursor_donut = size_donut - 1;
  if (cursor_donut < 0) cursor_donut = 0;
  lcd.clear();
  lcd.setCursor(cursor_donut, 0);
  lcd.print(iyi_eglenceler[l]);
  if (l == 2) {
    lcd.write((byte)2);
  }
  lcd.write((byte)1);
}

void kart_okuma() {
  lcd.clear();
  lcd.setCursor(1, 0);
  lcd.print("KART OKUNUYOR!");
}

void reset_state() {
  lcd.clear();
  lcd.setCursor(1, 0);
  lcd.print("KART OKUTUNUZ.");
}
void bekleyiniz() {
  String lutfen = "LUTFEN 3 SANIYE";
  String bekleyiniz = "BEKLEYINIZ";
  int cursor0, cursor1;
  cursor0 = (16 - lutfen.length()) / 2;
  cursor1 = (16 - bekleyiniz.length()) / 2;
  lcd.clear();
  lcd.setCursor(cursor0, 0);
  lcd.print(lutfen);
  lcd.setCursor(cursor1, 1);
  lcd.print(bekleyiniz);
}

void callback(char* topic, byte* message, unsigned int length) {
  messageTemp = "";
  Serial.print("Mesagge arrived on topic: ");
  Serial.println(topic);
  Serial.print("Message: ");
  for (int i = 0; i < length; ++i) {
    Serial.print((char)message[i]);
    messageTemp += (char)message[i];
  }
  Serial.println();
  //ekran kontrol
  /*if (messageTemp == 'y'){
    kart_donut(2);
    //role kontrol
    delay(1000);
    }
    else if (messageTemp == 'n'){
    int cursor0, cursor1;
    String gecersiz = "GECERSIZ";
    String kart = "KART";
    cursor0 = (16 - gecersiz.length()) / 2;
    cursor1 = (16 - kart.length()) / 2;
    lcd.clear();
    lcd.setCursor(cursor0, 0);
    lcd.print(gecersiz);
    lcd.setCursor(cursor1, 1);
    lcd.print(kart);
    delay(1000);
    }*/
  return messageTemp;
}

void reconnect() {
  Serial.println("In recconnect...");
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    if (client.connect("Kurabiye", mqtt_user, mqtt_pass)) {
      Serial.println("connected");
      client.subscribe("ARGETEKNO");
    }
    else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 2 sec");
      delay(2000);
    }
  }
}

void setup() {
  Serial.begin(9600);//initialize serial communication at 9600 bps
  //start the Ethernet connection and the server:
  Ethernet.begin(mac, ip);
  Serial.print("IP Address: ");
  Serial.println(Ethernet.localIP());
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);
  server.begin();
  pinMode(SS_Ethernet, OUTPUT);
  digitalWrite(SS_Ethernet, HIGH);
  digitalWrite(SS_PIN, HIGH);
  Ethernet.init(8);
  SPI.begin();
  rfid.PCD_Init();
  lcd.init();
  lcd.backlight();
  //lcd.noBacklight
  lcd.createChar(0, gulen_yuz);
  lcd.createChar(1, kalp);
  lcd.createChar(2, ss);
  hosgeldiniz_print();
  delay(1000);
  reset_state();
  pinMode(role, OUTPUT);
  digitalWrite(role, HIGH);
}
void read_rfid() {
  if ( ! rfid.PICC_IsNewCardPresent())
    return;

  if ( ! rfid.PICC_ReadCardSerial())
    return;
  Serial.println(F("-----------------------------------------"));
  Serial.print(F("PICC type: "));
  MFRC522::PICC_Type piccType = rfid.PICC_GetType(rfid.uid.sak);
  Serial.println(rfid.PICC_GetTypeName(piccType));

  // Check is the PICC of Classic MIFARE type
  if (piccType != MFRC522::PICC_TYPE_MIFARE_MINI &&
      piccType != MFRC522::PICC_TYPE_MIFARE_1K &&
      piccType != MFRC522::PICC_TYPE_MIFARE_4K) {
    Serial.println(F("Kartın etiketi hiçbir MIFARE klasiği ile uyumlu değil."));
    return;
  }

  /*if (rfid.uid.uidByte[0] != nuidPICC[0] ||
      rfid.uid.uidByte[1] != nuidPICC[1] ||
      rfid.uid.uidByte[2] != nuidPICC[2] ||
      rfid.uid.uidByte[3] != nuidPICC[3] )*/
  firsttime = millis();
  if (firsttime - secondtime > 3000) {
    kontrol = true;
    secondtime = millis();
  }
  else kontrol = false;
  if (kontrol == true) {
    Serial.println(F("Yeni bir kart tespit edildi."));

    for (byte i = 0; i < 4; i++) {
      nuidPICC[i] = rfid.uid.uidByte[i];
    }
    Serial.println(F("Okutulan kartın ID'si:"));
    Serial.print(F(" "));
    printDec(rfid.uid.uidByte, rfid.uid.size);
    ID = String(rfid.uid.uidByte[0]) +  String(rfid.uid.uidByte[1]) + String(rfid.uid.uidByte[2]) +  String(rfid.uid.uidByte[3]);
    ID.toCharArray(payload, 50);
    client.publish("D2Kare", payload);
    Serial.println();
    //kart_okuma();
  }
  else if (kontrol == false) {
    Serial.println("bekle");
    bekleyiniz();
  }
  rfid.PICC_HaltA();

  rfid.PCD_StopCrypto1();
}

void printDec(byte *buffer, byte bufferSize) {
  for (byte i = 0; i < bufferSize; i++) {
    Serial.print(' ');
    Serial.print(buffer[i], DEC);
  }
}
void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
  /*lcd.display();
    delay(800);
    lcd.noDisplay();
    delay(400);*/
  /*
    else if(firsttime - secondtime <= 3000){
    Serial.println(F("3sn kuralı"));

    }*/
  read_rfid();
  if (messageTemp == "y") {
    kart_donut(0);
    messageTemp = "";
    role_durum = 1;
  }
  else if (messageTemp == "n") {
    int cursor0, cursor1;
    String gecersiz = "GECERSIZ";
    String kart = "KART";
    cursor0 = (16 - gecersiz.length()) / 2;
    cursor1 = (16 - kart.length()) / 2;
    lcd.clear();
    lcd.setCursor(cursor0, 0);
    lcd.print("GECERSIZ");
    lcd.setCursor(cursor1, 1);
    lcd.print("KART");
    messageTemp = "";
  }
  if ((role_durum == 1) && (mil_new == 0)){
    Serial.println("Röle Durum 1 Oldu");
    mil_new = millis() + 3000;
  }
  if (mil_new > millis()){
    digitalWrite(role, LOW);
    Serial.println("Röle Açtı");
  }
  else {
    digitalWrite(role, HIGH);
    role_durum = 0;
    mil_new = 0;
   // Serial.println("Röle Kapattı");
  }
}
