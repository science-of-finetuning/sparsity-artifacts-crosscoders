function setupTokenTooltips() {
    // Create tooltip element if it doesn't exist
    if (!document.querySelector('.token-tooltip')) {
        const tooltip = document.createElement('div');
        tooltip.className = 'token-tooltip';
        tooltip.style.display = 'none';
        document.body.appendChild(tooltip);
    }
    
    // Add event listeners for all tokens
    document.querySelectorAll('.token').forEach(token => {
        token.addEventListener('mousemove', (e) => {
            const tooltip = document.querySelector('.token-tooltip');
            tooltip.textContent = token.dataset.tooltip;
            tooltip.style.display = 'block';
            tooltip.style.left = e.pageX + 10 + 'px';
            tooltip.style.top = e.pageY + 10 + 'px';
        });
        
        token.addEventListener('mouseleave', () => {
            const tooltip = document.querySelector('.token-tooltip');
            tooltip.style.display = 'none';
        });
    });
}
setupTokenTooltips(); 