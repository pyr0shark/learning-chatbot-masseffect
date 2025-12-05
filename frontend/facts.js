// Facts page functionality
class FactsPage {
    constructor() {
        this.approvedContainer = document.getElementById('approved-facts');
        this.rejectedContainer = document.getElementById('rejected-facts');
        this.approvedCount = document.getElementById('approved-count');
        this.rejectedCount = document.getElementById('rejected-count');
        
        // Load facts when page becomes active
        this.init();
    }
    
    init() {
        // Check if we're on the facts page
        const factsPage = document.getElementById('facts-page');
        if (factsPage && factsPage.classList.contains('active')) {
            this.loadFacts();
        }
        
        // Listen for page changes via hash changes (router navigation)
        window.addEventListener('hashchange', () => {
            const hash = window.location.hash.slice(1);
            if (hash === 'facts' || hash === '') {
                const factsPage = document.getElementById('facts-page');
                if (factsPage && factsPage.classList.contains('active')) {
                    this.loadFacts();
                }
            }
        });
        
        // Also listen for class changes on the facts page
        if (factsPage) {
            const observer = new MutationObserver((mutations) => {
                mutations.forEach((mutation) => {
                    if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                        if (factsPage.classList.contains('active')) {
                            this.loadFacts();
                        }
                    }
                });
            });
            observer.observe(factsPage, { attributes: true, attributeFilter: ['class'] });
        }
    }
    
    async loadFacts() {
        try {
            const response = await fetch('/api/facts/all');
            if (!response.ok) {
                throw new Error('Failed to load facts');
            }
            
            const data = await response.json();
            this.renderFacts(data.approved, data.rejected);
        } catch (error) {
            console.error('Error loading facts:', error);
            this.approvedContainer.innerHTML = '<div class="error-message">Error loading approved facts</div>';
            this.rejectedContainer.innerHTML = '<div class="error-message">Error loading rejected facts</div>';
        }
    }
    
    renderFacts(approved, rejected) {
        // Update counts
        this.approvedCount.textContent = approved.length;
        this.rejectedCount.textContent = rejected.length;
        
        // Render approved facts
        if (approved.length === 0) {
            this.approvedContainer.innerHTML = '<div class="empty-message">No approved facts yet</div>';
        } else {
            this.approvedContainer.innerHTML = approved.map(fact => this.renderFactCard(fact, 'approved')).join('');
        }
        
        // Render rejected facts
        if (rejected.length === 0) {
            this.rejectedContainer.innerHTML = '<div class="empty-message">No rejected facts yet</div>';
        } else {
            this.rejectedContainer.innerHTML = rejected.map(fact => this.renderFactCard(fact, 'rejected')).join('');
        }
    }
    
    renderFactCard(fact, status) {
        const date = fact.approved_at || fact.rejected_at || fact.created_at;
        const dateStr = date ? new Date(date).toLocaleString() : 'Unknown date';
        
        return `
            <div class="fact-card ${status}">
                <div class="fact-card-header">
                    <span class="fact-status-badge ${status}">${status === 'approved' ? '✅ Approved' : '❌ Rejected'}</span>
                    <span class="fact-date">${dateStr}</span>
                </div>
                <div class="fact-card-content">
                    <p>${this.escapeHtml(fact.fact)}</p>
                </div>
            </div>
        `;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize facts page when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new FactsPage();
});

