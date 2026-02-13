CC = gcc
INCLUDE_DIRS = include
SRC_DIR = src
OBJ_DIR = obj

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

CFLAGS =  -O3 -march=native -ffast-math
LIBDIR = lib
LFLAGS = -lm -lopenblas

ifeq ($(OS),Windows_NT)
    TARGET = mnist.exe
    RMD = rmdir /s /q
    RMF = del /q
    MKDIR = mkdir
else
    TARGET = mnist
    RMD = rm -rf
    RMF = rm -f
    MKDIR = mkdir -p
endif

all: $(TARGET)


$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -L $(LIBDIR)  $(LFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) -I$(INCLUDE_DIRS) -c $< -o $@

$(OBJ_DIR):
	$(MKDIR) $(OBJ_DIR)




run : $(TARGET)
	./$(TARGET)
clean:
	$(RMD) $(OBJ_DIR)
	$(RMF) $(TARGET)


.PHONY: all clean run
