function getQueryParams() {
    const params = {};
    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);

    urlParams.forEach((value, key) => {
        params[key] = value;
    });
    return params;
}

function handleError(error, message) {
    console.error(message, error);
    showErrorPopup(`${message} ${error.message}`);
}

function scrollChatBoxToBottom() {
    const chatbox = document.getElementById('chatbox');
    chatbox.scrollTop = chatbox.scrollHeight;
}

const showErrorPopup = (message) => {
    const errorPopup = document.getElementById('error-popup');
    const errorMessageText = document.getElementById('error-popup-message-text');

    errorPopup.classList.remove('hidden');
    errorMessageText.innerHTML = message;

    // Auto-hide after 5 seconds
    setTimeout(() => {
        errorPopup.classList.add('hidden');
    }, 5000);
};

function throwOrJson(response) {
    if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return response.json();
}

    // Function to check button width and replace text with icon if needed
function updateSendButton() {
    const sendButton = document.getElementById('send-button');
    if (sendButton.offsetWidth < 60) { // Adjust the width as needed
        sendButton.innerHTML = '<i class="fas fa-arrow-right"></i>';
    } else {
        sendButton.innerHTML = 'Send';
    }
}