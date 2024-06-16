function getQueryParams() {
    const params = {};
    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);

    urlParams.forEach((value, key) => {
        params[key] = value;
    });
    return params;
}

function getLoadingIndicator() {
    const loadingIndicator = document.createElement('div');
    loadingIndicator.className = 'loading-indicator';
    loadingIndicator.innerHTML = '<div class="loading-spinner"></div>';
    return loadingIndicator;
}

function getUserMessageBox(message) {
    const userMessageBox = document.createElement('div');
    userMessageBox.className = 'user-message-box';
    userMessageBox.innerHTML = `<i class="fa fa-user"></i><b>user:</b>`;
    userMessageBox.innerHTML += `${marked.parse(message)}</br>`;

    // Render user message copy button
    const userMessageCopyLink = document.createElement('a');
    userMessageCopyLink.title = "Copy"
    userMessageCopyLink.className = "chatbox-message-link";
    userMessageCopyLink.innerHTML = `<i class="fa fa-copy" style="">`;
    userMessageCopyLink.addEventListener('click', (event) => {
        event.preventDefault();
        navigator.clipboard.writeText(message)
    })
    userMessageBox.appendChild(userMessageCopyLink);
    hljs.highlightAll();
    return userMessageBox;
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