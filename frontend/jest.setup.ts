import "@testing-library/jest-dom";

// Every suite starts with no fetch stub in place: a test that forgets to mock
// one must fail loudly rather than reach a real network.
beforeEach(() => {
  global.fetch = jest.fn(() => {
    throw new Error("unmocked fetch — stub it in the test");
  }) as unknown as typeof fetch;
});

afterEach(() => {
  jest.restoreAllMocks();
});
