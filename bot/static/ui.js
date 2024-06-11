function handleError(error, message) {
    console.error(message, error);
    showErrorPopup(`${message} ${error.message}`);
}

function scrollChatBoxToBottom() {
    const chatbox = document.getElementById('chatbox');
    chatbox.scrollTop = chatbox.scrollHeight;
}

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