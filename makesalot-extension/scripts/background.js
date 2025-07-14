chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === "GET_SYMBOL_FROM_TAB") {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const tab = tabs[0];
      const url = tab.url;
      let symbol = "";
      let api = "alpha_vantage";

      if (url.includes("finance.yahoo.com")) {
        const match = url.match(/quote\/([A-Z.]+)/);
        if (match) symbol = match[1];
        api = "yahoo_finance";
      } else if (url.includes("tradingview.com/symbols/")) {
        const match = url.match(/symbols\/([^\/]+)/);
        if (match) symbol = match[1].toUpperCase();
        api = "polygon";
      } else if (url.includes("investing.com")) {
        const match = url.match(/equities\/([^\/]+)/);
        if (match) symbol = match[1].replace("-", "_").toUpperCase();
        api = "polygon";
      }

      // Atualiza badge (ícone)
      if (symbol) {
        chrome.action.setBadgeText({ text: symbol.substring(0, 4), tabId: tab.id });
        chrome.action.setBadgeBackgroundColor({ color: "#22c55e", tabId: tab.id });
      } else {
        chrome.action.setBadgeText({ text: "", tabId: tab.id });
      }

      sendResponse({ symbol, api });
    });

    return true;
  }
});
