CC = gcc
INCLUDE_DIRS = include
SRC_DIR = src
OBJ_DIR = obj
TARGET = main.exe

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

CFLAGS = -I$(INCLUDE_DIRS) -O3 -march=native -ffast-math

LFLAGS = -lm -lopenblas

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC)   -o $@ $^ $(LFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR):
	@if not exist $(OBJ_DIR) mkdir $(OBJ_DIR)

clean:
	@if exist $(OBJ_DIR) rmdir /s /q $(OBJ_DIR)
	@if exist $(TARGET) del /q $(TARGET)


.PHONY: all clean
