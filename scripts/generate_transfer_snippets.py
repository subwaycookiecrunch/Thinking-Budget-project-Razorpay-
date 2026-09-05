#!/usr/bin/env python3
"""
scripts/generate_transfer_snippets.py
======================================
Generate realistic code snippets for the 5 transfer episodes so that
when the model calls `read_file` on a transfer-domain file, it gets
real code instead of '// [source code not available]'.

These are synthetic but REALISTIC code snippets that:
- Match the file language and component
- Include the described vulnerability pattern for buggy files
- Include normal/safe code for non-buggy files
- Are long enough to look real (50-200 lines)

Usage:
    python scripts/generate_transfer_snippets.py
    # Output: merges into code_review_env/data/code_snippets.json
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EPISODES_PATH = ROOT / "data" / "transfer_episodes.json"
SNIPPETS_PATH = ROOT / "code_review_env" / "data" / "code_snippets.json"

# ── Code templates for each transfer episode ──────────────────────────

TRANSFER_SNIPPETS = {
    # ═══════════════════════════════════════════════════════════════
    # TR-CR-001: Payment refactor — race condition
    # ═══════════════════════════════════════════════════════════════
    "src/api/orders/create.ts": '''import { PaymentService } from "../payments/charge";
import { OrderRepo } from "../../db/repo/orders";
import { acquireLock } from "../../utils/lock";

interface CreateOrderRequest {
  userId: string;
  items: Array<{ productId: string; quantity: number; price: number }>;
  paymentMethod: string;
}

export async function createOrder(req: CreateOrderRequest) {
  const total = req.items.reduce((sum, i) => sum + i.price * i.quantity, 0);

  // BUG: Race condition — balance is read here...
  const balance = await PaymentService.getBalance(req.userId);
  if (balance < total) {
    throw new Error("Insufficient balance");
  }

  // ...but another request can charge between the read above
  // and the charge below, causing a double-charge
  const chargeResult = await PaymentService.charge({
    userId: req.userId,
    amount: total,
    // NOTE: idempotency token was removed during refactor
  });

  if (!chargeResult.success) {
    throw new Error("Payment failed");
  }

  const order = await OrderRepo.create({
    userId: req.userId,
    items: req.items,
    total,
    paymentId: chargeResult.paymentId,
    status: "confirmed",
  });

  return { orderId: order.id, total, paymentId: chargeResult.paymentId };
}
''',

    "src/api/orders/list.ts": '''import { OrderRepo } from "../../db/repo/orders";

interface ListOrdersQuery {
  userId: string;
  page?: number;
  limit?: number;
}

export async function listOrders(query: ListOrdersQuery) {
  const page = query.page || 1;
  const limit = Math.min(query.limit || 20, 100);
  const offset = (page - 1) * limit;

  const orders = await OrderRepo.findByUser(query.userId, { limit, offset });
  const total = await OrderRepo.countByUser(query.userId);

  return {
    orders,
    pagination: { page, limit, total, pages: Math.ceil(total / limit) },
  };
}
''',

    "src/api/orders/types.ts": '''export interface Order {
  id: string;
  userId: string;
  items: OrderItem[];
  total: number;
  status: "pending" | "confirmed" | "shipped" | "cancelled";
  paymentId?: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface OrderItem {
  productId: string;
  quantity: number;
  price: number;
}
''',

    "src/api/payments/charge.ts": '''import { PaymentGateway } from "./gateway";
import { PaymentRepo } from "../../db/repo/payments";

interface ChargeRequest {
  userId: string;
  amount: number;
  // NOTE: idempotencyToken was here but removed during refactor
}

export class PaymentService {
  static async getBalance(userId: string): Promise<number> {
    const account = await PaymentRepo.getAccount(userId);
    return account?.balance || 0;
  }

  static async charge(req: ChargeRequest) {
    // BUG: No idempotency check — duplicate charges possible
    // Previously had: if (await PaymentRepo.findByToken(req.idempotencyToken)) return existing;
    const result = await PaymentGateway.processPayment({
      userId: req.userId,
      amount: req.amount,
      currency: "USD",
    });

    if (result.success) {
      await PaymentRepo.deductBalance(req.userId, req.amount);
      await PaymentRepo.recordPayment({
        userId: req.userId,
        amount: req.amount,
        gatewayId: result.transactionId,
        status: "completed",
      });
    }

    return {
      success: result.success,
      paymentId: result.transactionId,
    };
  }
}
''',

    "src/api/payments/refund.ts": '''import { PaymentGateway } from "./gateway";
import { PaymentRepo } from "../../db/repo/payments";

export async function processRefund(paymentId: string, reason: string) {
  const payment = await PaymentRepo.findById(paymentId);
  if (!payment) throw new Error("Payment not found");
  if (payment.status === "refunded") throw new Error("Already refunded");

  const result = await PaymentGateway.refund(payment.gatewayId, payment.amount);
  if (result.success) {
    await PaymentRepo.updateStatus(paymentId, "refunded");
    await PaymentRepo.addBalance(payment.userId, payment.amount);
  }
  return result;
}
''',

    "src/api/payments/types.ts": '''export interface Payment {
  id: string;
  userId: string;
  amount: number;
  status: "pending" | "completed" | "refunded";
  gatewayId: string;
}

export type PaymentMethod = "card" | "bank" | "wallet";
''',

    "src/utils/lock.ts": '''import Redis from "ioredis";

const redis = new Redis(process.env.REDIS_URL || "redis://localhost:6379");

export async function acquireLock(
  key: string,
  ttlMs: number = 5000
): Promise<string | null> {
  const token = crypto.randomUUID();
  const result = await redis.set(
    `lock:${key}`, token, "PX", ttlMs, "NX"
  );
  return result === "OK" ? token : null;
}

export async function releaseLock(key: string, token: string): Promise<boolean> {
  const script = `
    if redis.call("get", KEYS[1]) == ARGV[1] then
      return redis.call("del", KEYS[1])
    else
      return 0
    end
  `;
  const result = await redis.eval(script, 1, `lock:${key}`, token);
  return result === 1;
}
''',

    "src/db/schema.sql": '''-- Migration: add idempotency support
ALTER TABLE payments ADD COLUMN idempotency_key VARCHAR(255) UNIQUE;
CREATE INDEX idx_payments_idempotency ON payments(idempotency_key);
ALTER TABLE payments ADD COLUMN created_at TIMESTAMP DEFAULT NOW();
''',

    "src/api/orders/__tests__/create.test.ts": '''import { createOrder } from "../create";
import { PaymentService } from "../../payments/charge";

jest.mock("../../payments/charge");

describe("createOrder", () => {
  it("should create order with valid payment", async () => {
    (PaymentService.getBalance as jest.Mock).mockResolvedValue(1000);
    (PaymentService.charge as jest.Mock).mockResolvedValue({
      success: true, paymentId: "pay_123"
    });
    const result = await createOrder({
      userId: "user_1",
      items: [{ productId: "p1", quantity: 1, price: 100 }],
      paymentMethod: "card",
    });
    expect(result.orderId).toBeDefined();
  });

  it("should reject insufficient balance", async () => {
    (PaymentService.getBalance as jest.Mock).mockResolvedValue(50);
    await expect(createOrder({
      userId: "user_1",
      items: [{ productId: "p1", quantity: 1, price: 100 }],
      paymentMethod: "card",
    })).rejects.toThrow("Insufficient");
  });

  // NOTE: No test for concurrent requests (race condition)
});
''',

    "src/api/payments/__tests__/charge.test.ts": '''import { PaymentService } from "../charge";

describe("PaymentService.charge", () => {
  it("should process payment", async () => {
    const result = await PaymentService.charge({
      userId: "user_1", amount: 100,
    });
    expect(result.success).toBe(true);
  });
  // NOTE: No test for duplicate charges without idempotency token
});
''',

    # ═══════════════════════════════════════════════════════════════
    # TR-CR-002: Auth middleware — path-prefix bypass
    # ═══════════════════════════════════════════════════════════════
    "src/auth/middleware.ts": '''import { verifyJWT } from "./jwt";
import { Request, Response, NextFunction } from "express";

const PUBLIC_PATHS = ["/health", "/api/public"];

export function authMiddleware(req: Request, res: Response, next: NextFunction) {
  // BUG: Uses startsWith instead of exact match
  // req.path = "/health/admin" matches "/health" prefix
  // This allows unauthenticated access to /health/* routes
  const isPublic = PUBLIC_PATHS.some(p => req.path.startsWith(p));

  if (isPublic) {
    return next();  // Skip auth for "public" paths
  }

  const token = req.headers.authorization?.replace("Bearer ", "");
  if (!token) {
    return res.status(401).json({ error: "No token provided" });
  }

  try {
    const payload = verifyJWT(token);
    req.user = payload;
    next();
  } catch (err) {
    res.status(401).json({ error: "Invalid token" });
  }
}
''',

    "src/auth/jwt.ts": '''import jwt from "jsonwebtoken";

const SECRET = process.env.JWT_SECRET || "dev-secret";
const EXPIRY = "24h";

export function signJWT(payload: Record<string, any>): string {
  return jwt.sign(payload, SECRET, { expiresIn: EXPIRY });
}

export function verifyJWT(token: string): Record<string, any> {
  return jwt.verify(token, SECRET) as Record<string, any>;
}
''',

    "src/auth/session.ts": '''// DEPRECATED: Session-based auth (being replaced by JWT)
// This file is scheduled for deletion.

import session from "express-session";

export const sessionMiddleware = session({
  secret: process.env.SESSION_SECRET || "dev",
  resave: false,
  saveUninitialized: false,
  cookie: { secure: process.env.NODE_ENV === "production" },
});
''',

    "src/auth/types.ts": '''export interface User {
  id: string;
  email: string;
  role: "admin" | "user" | "viewer";
  orgId: string;
}

declare global {
  namespace Express {
    interface Request {
      user?: User;
    }
  }
}
''',

    # ═══════════════════════════════════════════════════════════════
    # TR-CR-003: ML training pipeline — reproducibility regression
    # ═══════════════════════════════════════════════════════════════
    "src/train.py": '''import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model import TransformerModel
from src.data import load_dataset
from src.optimizer import build_optimizer

def train(config):
    # BUG: torch.compile is called BEFORE seed setting
    # This injects non-determinism into the compilation cache
    model = TransformerModel(config)
    model = torch.compile(model)  # <-- This should be AFTER seed setting

    # Seed setting happens too late
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    optimizer = build_optimizer(model, config)
    dataset = load_dataset(config.data_path)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}")

        if (epoch + 1) % config.save_every == 0:
            torch.save(model.state_dict(), f"checkpoints/epoch_{epoch}.pt")

if __name__ == "__main__":
    from configs import load_config
    config = load_config("configs/baseline.yaml")
    train(config)
''',

    "src/data.py": '''import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, path, max_len=512):
        with open(path) as f:
            self.texts = f.readlines()
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx].strip()[:self.max_len]

def load_dataset(path):
    return TextDataset(path)
''',

    "src/model.py": '''import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.ff_size,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, x):
        h = self.embed(x)
        h = self.encoder(h)
        return self.head(h)
''',

    "src/checkpoint.py": '''import torch
import os

def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]
''',

    # ═══════════════════════════════════════════════════════════════
    # TR-CR-004: Frontend — stale closure
    # ═══════════════════════════════════════════════════════════════
    "src/components/Dashboard.tsx": '''import React, { useCallback, useMemo, useEffect, useState } from "react";
import { useUser } from "../hooks/useUser";
import { useOrg } from "../hooks/useOrg";
import { fetchDashboardData } from "../api/dashboard";

export const Dashboard: React.FC = () => {
  const { userId } = useUser();
  const { orgId, switchOrg } = useOrg();
  const [data, setData] = useState(null);

  // BUG: useCallback captures userId at initial render
  // When org switches, userId updates but this callback still
  // references the OLD userId from the closure
  const loadData = useCallback(async () => {
    // userId is STALE here after org switch
    const result = await fetchDashboardData(userId, orgId);
    setData(result);
  }, []);  // <-- Missing userId in dependency array

  useEffect(() => {
    loadData();
  }, [orgId]);  // Fires on org switch but loadData has stale userId

  const metrics = useMemo(() => {
    if (!data) return [];
    return data.metrics.map(m => ({
      ...m,
      formatted: `${m.value.toLocaleString()} ${m.unit}`,
    }));
  }, [data]);

  return (
    <div className="dashboard">
      <h1>Dashboard — {orgId}</h1>
      <div className="metrics-grid">
        {metrics.map(m => (
          <div key={m.id} className="metric-card">
            <span className="label">{m.label}</span>
            <span className="value">{m.formatted}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
''',

    "src/components/Sidebar.tsx": '''import React from "react";

interface SidebarProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, onTabChange }) => {
  const tabs = ["dashboard", "orders", "products", "settings"];

  return (
    <nav className="sidebar">
      {tabs.map(tab => (
        <button
          key={tab}
          className={`tab ${activeTab === tab ? "active" : ""}`}
          onClick={() => onTabChange(tab)}
        >
          {tab.charAt(0).toUpperCase() + tab.slice(1)}
        </button>
      ))}
    </nav>
  );
};
''',

    # ═══════════════════════════════════════════════════════════════
    # TR-CR-005: DB query optimization — tenant leak
    # ═══════════════════════════════════════════════════════════════
    "src/db/queries/users.sql": '''-- Batched user lookup for dashboard
-- BUG: Missing WHERE tenant_id = $1
-- This query returns ALL users across ALL tenants
-- Previous version had: WHERE tenant_id = $1 AND id = ANY($2)

SELECT
  u.id,
  u.email,
  u.display_name,
  u.role,
  u.created_at,
  u.last_login,
  COUNT(o.id) as order_count,
  COALESCE(SUM(o.total), 0) as total_spent
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
WHERE u.id = ANY($1)  -- <-- Missing: AND u.tenant_id = $2
GROUP BY u.id
ORDER BY u.last_login DESC;
''',

    "src/db/queries/orders.sql": '''-- Batched order lookup with tenant filter (CORRECT)
SELECT
  o.id, o.user_id, o.total, o.status, o.created_at
FROM orders o
WHERE o.tenant_id = $1  -- Tenant filter present
  AND o.user_id = ANY($2)
ORDER BY o.created_at DESC;
''',

    "src/db/queries/products.sql": '''-- Product catalog with tenant filter (CORRECT)
SELECT
  p.id, p.name, p.price, p.category, p.stock
FROM products p
WHERE p.tenant_id = $1  -- Tenant filter present
  AND p.active = true
ORDER BY p.name;
''',

    "src/db/repo/users.ts": '''import { query } from "../connection";
import { readFileSync } from "fs";

const USER_QUERY = readFileSync("src/db/queries/users.sql", "utf-8");

export class UserRepo {
  static async findByIds(userIds: string[]) {
    // Trusts the SQL query for tenant isolation
    // But the query is missing the tenant_id filter!
    const result = await query(USER_QUERY, [userIds]);
    return result.rows;
  }

  static async findById(userId: string) {
    const result = await query(
      "SELECT * FROM users WHERE id = $1",
      [userId]
    );
    return result.rows[0] || null;
  }
}
''',
}


def main():
    # Load existing snippets
    if SNIPPETS_PATH.exists():
        with open(SNIPPETS_PATH) as f:
            snippets = json.load(f)
    else:
        snippets = {}

    added = 0
    for fpath, code in TRANSFER_SNIPPETS.items():
        if fpath not in snippets or snippets[fpath] == "// [source code not available]":
            snippets[fpath] = code
            added += 1

    # Also add simple stubs for remaining transfer files without custom snippets
    with open(EPISODES_PATH) as f:
        episodes = json.load(f)

    for ep in episodes:
        for f in ep["files"]:
            fpath = f["file"]
            if fpath not in snippets:
                lang = f.get("language", "")
                summary = f.get("summary", "")
                is_test = f.get("is_test", False)
                label = f["label"]

                if is_test:
                    code = f'// Test file: {fpath}\n// {summary}\n\ndescribe("{fpath}", () => {{\n  it("should pass", () => {{\n    expect(true).toBe(true);\n  }});\n}});\n'
                elif lang in ("Markdown", "JSON", "YAML", "Env", "Shell", "CSS", "Make"):
                    code = f"// {fpath}\n// {summary}\n"
                else:
                    code = f"// {fpath}\n// {summary}\n// Language: {lang}\n"

                snippets[fpath] = code
                added += 1

    with open(SNIPPETS_PATH, "w") as f:
        json.dump(snippets, f, indent=2)

    alt_path = ROOT / "data" / "code_snippets.json"
    if alt_path.parent.exists():
        with open(alt_path, "w") as f:
            json.dump(snippets, f, indent=2)

    print(f"✅ Added {added} transfer-domain code snippets to {SNIPPETS_PATH} and {alt_path}")
    print(f"   Total snippets: {len(snippets)}")


if __name__ == "__main__":
    main()
