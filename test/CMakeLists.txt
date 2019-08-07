enable_testing()
set(PROJECT_TEST_NAME ${PROJECT_NAME}_test)
add_executable( ${PROJECT_TEST_NAME}
	runTests.cpp
	testImuIntegration.cpp
	#testImuError.cpp
)
target_link_libraries(${PROJECT_TEST_NAME}
	${GTEST_LIBRARY}
	pthread
)
add_test(test ${PROJECT_TEST_NAME})