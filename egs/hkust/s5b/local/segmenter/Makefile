JAVA_COMPILER=javac
SOURCE_DIR=.
BIN_DIR=.
JAR_FILE=ChiUtf8Segmenter.jar


SOURCES = $(wildcard $(SOURCE_DIR)/*.java)
CLASSES = $(patsubst $(SOURCE_DIR)/%.java, $(BIN_DIR)/%.class, $(SOURCES))

all: $(JAR_FILE)

$(JAR_FILE): $(CLASSES)
	jar -cmf MANIFEST.MF $(JAR_FILE) -C $(BIN_DIR) .
	chmod +x $(JAR_FILE)

$(BIN_DIR)/%.class: $(SOURCE_DIR)/%.java
	$(JAVA_COMPILER) -d $(BIN_DIR) $(SOURCE_DIR)/*.java

clean:
	rm -f $(BIN_DIR)/*.class
	rm -f $(JAR_FILE)

