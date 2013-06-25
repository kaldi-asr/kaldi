JAVA_COMPILER=javac
SOURCE_DIR=src
BIN_DIR=bin
JAR_FILE=java-online-audio-client.jar


SOURCES = $(wildcard $(SOURCE_DIR)/*.java)
CLASSES = $(patsubst $(SOURCE_DIR)/%.java, $(BIN_DIR)/%.class, $(SOURCES))

all: $(JAR_FILE)

$(JAR_FILE): $(CLASSES)
	jar -cmf MANIFEST.MF $(JAR_FILE) -C $(BIN_DIR) .
	chmod +x $(JAR_FILE)

$(BIN_DIR)/%.class: $(SOURCE_DIR)/%.java
	javac -d $(BIN_DIR) $(SOURCE_DIR)/*.java

clean:
	rm -f bin/*.class
	rm -f $(JAR_FILE)
