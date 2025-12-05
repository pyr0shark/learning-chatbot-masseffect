// Simple client-side router
class Router {
    constructor() {
        this.currentPage = 'chat';
        this.init();
    }
    
    init() {
        // Get initial page from hash or default to chat
        const hash = window.location.hash.slice(1);
        if (hash) {
            this.navigateTo(hash);
        } else {
            this.navigateTo('chat');
        }
        
        // Listen for hash changes
        window.addEventListener('hashchange', () => {
            const hash = window.location.hash.slice(1);
            this.navigateTo(hash || 'chat');
        });
        
        // Set up navigation buttons
        document.querySelectorAll('.nav-button').forEach(button => {
            button.addEventListener('click', (e) => {
                const page = e.currentTarget.dataset.page;
                this.navigateTo(page);
            });
        });
    }
    
    navigateTo(page) {
        // Hide all pages
        document.querySelectorAll('.page').forEach(pageEl => {
            pageEl.classList.remove('active');
        });
        
        // Show target page
        const targetPage = document.getElementById(`${page}-page`);
        if (targetPage) {
            targetPage.classList.add('active');
        }
        
        // Update nav buttons
        document.querySelectorAll('.nav-button').forEach(button => {
            if (button.dataset.page === page) {
                button.classList.add('active');
            } else {
                button.classList.remove('active');
            }
        });
        
        // Update hash (without triggering hashchange)
        if (window.location.hash !== `#${page}`) {
            window.history.pushState(null, '', `#${page}`);
        }
        
        this.currentPage = page;
    }
}

// Initialize router when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new Router();
});

