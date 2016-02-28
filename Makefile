CC=g++
CFLAGS=`pkg-config --cflags --libs opencv`

SRCS= detect.cc describe.cc match.cc utils.cc
OBJS=${SRCS:.cc=.o}
EXA= example.cc

match: ${OBJS}
	${CC} ${CFLAGS} -o match ${OBJS}

%.o : %.cc
	${CC}  -c $< -o $@

example: ${EXA}
	${CC} ${CFLAGS} -o $@ ${EXA}
