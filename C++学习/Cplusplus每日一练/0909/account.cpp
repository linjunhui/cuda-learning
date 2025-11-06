
#include <iostream>

class Account {
    private:
        int id;
        float balance;

    public:
        Account(int id, float balance) {
            this->id = id;
            this->balance = balance;
        }

        // 存款
        void deposit(float amount) {
            this->balance += amount;
        }

        // 取款
        void withdraw(float amount) {
            if (this->balance < amount) {
                std::cout << "余额不足" << std::endl;
                return;
            }
            this->balance -= amount;
        }

        float getBalance() {
            return this->balance;

        }

};

int main() {
    Account account(100, 100.5);

    account.deposit(100);

    std::cout << account.getBalance() << std::endl;
}