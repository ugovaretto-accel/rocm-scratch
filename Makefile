HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --platform)
HIPCC=$(HIP_PATH)/bin/hipcc

#SOURCES=threadfence_system.cpp

all: threadfence_system threadfence_system2

threadfence_system: threadfence_system.cpp
	$(HIPCC) $(CXXFLAGS) threadfence_system.cpp -o $@.out

threadfence_system2: threadfence_system.cpp
	$(HIPCC) $(CXXFLAGS) threadfence_system2.cpp -o $@.out

clean:
	rm -f *.o *.out 
