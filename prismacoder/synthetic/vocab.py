"""12 domain archetypes with entity templates (report 9.4).

Each archetype parameterizes the grammar sampler:
- entity templates: plausible models + scalar fields + relation hints
- enums: domain status vocabularies
- key styles, soft-delete, multi-tenancy and audit-timestamp toggles are
  applied by the sampler per schema (not baked into the archetype).

Field grammar notes:
- every entity template gets an implicit `id` key field (style chosen by the sampler)
- `rel` entries describe (cardinality, target, name): "1m" = this has many target,
  "11" = optional single, "mn" = many-to-many, "self" = self-relation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type: str  # String | Int | Float | Boolean | DateTime | Decimal | Json | BigInt
    optional: bool = False
    unique: bool = False
    default: str | None = None  # rendered verbatim, e.g. "now()" / "false" / '"PENDING"'


@dataclass(frozen=True)
class RelSpec:
    cardinality: str  # "1m" | "11" | "mn" | "self"
    target: str
    name: str  # field name on this side (sampler dedups the other side)
    target_name: str = ""  # field name on the target side ("" = derive)


@dataclass(frozen=True)
class EntitySpec:
    name: str
    fields: tuple[FieldSpec, ...]
    rels: tuple[RelSpec, ...] = ()


@dataclass(frozen=True)
class EnumSpec:
    name: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class DomainSpec:
    key: str  # one of prismacoder.constants.DOMAINS
    label: str
    entities: tuple[EntitySpec, ...]
    enums: tuple[EnumSpec, ...] = ()
    tenant_field: str | None = None  # "organizationId" -> multi-tenancy toggle


# --- shared building blocks --------------------------------------------------
AUDIT = (
    FieldSpec("createdAt", "DateTime", default="now()"),
    FieldSpec("updatedAt", "DateTime", default="now()"),
)
SOFT_DELETE = FieldSpec("deletedAt", "DateTime", optional=True)

USER = EntitySpec(
    "User",
    (
        FieldSpec("email", "String", unique=True),
        FieldSpec("name", "String", optional=True),
        FieldSpec("role", "String", default='"MEMBER"'),
        FieldSpec("isActive", "Boolean", default="true"),
        *AUDIT,
    ),
)

DOMAINS_SPECS: tuple[DomainSpec, ...] = (
    DomainSpec(
        "ecommerce",
        "E-commerce storefront",
        (
            EntitySpec(
                "User",
                (
                    FieldSpec("email", "String", unique=True),
                    FieldSpec("name", "String", optional=True),
                    FieldSpec("isActive", "Boolean", default="true"),
                    *AUDIT,
                ),
                (RelSpec("1m", "Order", "orders"), RelSpec("1m", "Review", "reviews")),
            ),
            EntitySpec(
                "Product",
                (
                    FieldSpec("sku", "String", unique=True),
                    FieldSpec("title", "String"),
                    FieldSpec("priceCents", "Int"),
                    FieldSpec("stock", "Int", default="0"),
                    FieldSpec("description", "String", optional=True),
                    *AUDIT,
                ),
                (
                    RelSpec("1m", "OrderItem", "items"),
                    RelSpec("mn", "Category", "categories"),
                    RelSpec("1m", "Review", "reviews"),
                ),
            ),
            EntitySpec(
                "Category",
                (FieldSpec("slug", "String", unique=True), FieldSpec("name", "String"), *AUDIT),
                (),
            ),
            EntitySpec(
                "Order",
                (
                    FieldSpec("status", "String", default='"PENDING"'),
                    FieldSpec("totalCents", "Int", default="0"),
                    FieldSpec("placedAt", "DateTime", default="now()"),
                    *AUDIT,
                ),
                (RelSpec("1m", "OrderItem", "items"),),
            ),
            EntitySpec(
                "OrderItem",
                (FieldSpec("quantity", "Int", default="1"), FieldSpec("unitPriceCents", "Int"), *AUDIT),
                (),
            ),
            EntitySpec(
                "Review", (FieldSpec("rating", "Int"), FieldSpec("body", "String", optional=True), *AUDIT), ()
            ),
        ),
        (EnumSpec("OrderStatus", ("PENDING", "PAID", "SHIPPED", "DELIVERED", "CANCELLED")),),
    ),
    DomainSpec(
        "saas_billing",
        "SaaS billing",
        (
            EntitySpec(
                "Organization",
                (
                    FieldSpec("slug", "String", unique=True),
                    FieldSpec("name", "String"),
                    FieldSpec("plan", "String", default='"FREE"'),
                    *AUDIT,
                ),
                (
                    RelSpec("1m", "Subscription", "subscriptions"),
                    RelSpec("1m", "Invoice", "invoices"),
                    RelSpec("1m", "Member", "members"),
                ),
            ),
            EntitySpec(
                "Member",
                (
                    FieldSpec("role", "String", default='"MEMBER"'),
                    FieldSpec("invitedEmail", "String", optional=True),
                    *AUDIT,
                ),
                (),
            ),
            EntitySpec(
                "Subscription",
                (
                    FieldSpec("status", "String", default='"ACTIVE"'),
                    FieldSpec("currentPeriodEnd", "DateTime", optional=True),
                    FieldSpec("seats", "Int", default="1"),
                    *AUDIT,
                ),
                (RelSpec("1m", "Invoice", "invoices"),),
            ),
            EntitySpec(
                "Invoice",
                (
                    FieldSpec("number", "String", unique=True),
                    FieldSpec("amountCents", "Int"),
                    FieldSpec("paidAt", "DateTime", optional=True),
                    *AUDIT,
                ),
                (RelSpec("1m", "InvoiceLine", "lines"),),
            ),
            EntitySpec(
                "InvoiceLine",
                (
                    FieldSpec("description", "String"),
                    FieldSpec("quantity", "Int", default="1"),
                    FieldSpec("unitAmountCents", "Int"),
                    *AUDIT,
                ),
                (),
            ),
            EntitySpec(
                "UsageRecord",
                (
                    FieldSpec("metric", "String"),
                    FieldSpec("quantity", "Int"),
                    FieldSpec("recordedAt", "DateTime", default="now()"),
                ),
                (),
            ),
        ),
        (
            EnumSpec("Plan", ("FREE", "PRO", "BUSINESS", "ENTERPRISE")),
            EnumSpec("InvoiceStatus", ("DRAFT", "OPEN", "PAID", "VOID")),
        ),
    ),
    DomainSpec(
        "social",
        "Social network",
        (
            EntitySpec(
                "User",
                (
                    FieldSpec("username", "String", unique=True),
                    FieldSpec("displayName", "String", optional=True),
                    FieldSpec("bio", "String", optional=True),
                    *AUDIT,
                ),
                (
                    RelSpec("1m", "Post", "posts"),
                    RelSpec("1m", "Follow", "following"),
                    RelSpec("1m", "Like", "likes"),
                ),
            ),
            EntitySpec(
                "Post",
                (
                    FieldSpec("content", "String"),
                    FieldSpec("visibility", "String", default='"PUBLIC"'),
                    FieldSpec("publishedAt", "DateTime", optional=True),
                    *AUDIT,
                ),
                (RelSpec("1m", "Like", "likes"), RelSpec("1m", "Comment", "comments")),
            ),
            EntitySpec(
                "Comment",
                (FieldSpec("body", "String"), *AUDIT),
                (RelSpec("self", "Comment", "parent", "replies"),),
            ),
            EntitySpec("Like", (FieldSpec("createdAt", "DateTime", default="now()"),), ()),
            EntitySpec("Follow", (FieldSpec("createdAt", "DateTime", default="now()"),), ()),
        ),
        (EnumSpec("Visibility", ("PUBLIC", "FOLLOWERS", "PRIVATE")),),
    ),
    DomainSpec(
        "cms",
        "Content management",
        (
            EntitySpec(
                "Author",
                (FieldSpec("email", "String", unique=True), FieldSpec("name", "String"), *AUDIT),
                (RelSpec("1m", "Article", "articles"),),
            ),
            EntitySpec(
                "Article",
                (
                    FieldSpec("slug", "String", unique=True),
                    FieldSpec("title", "String"),
                    FieldSpec("body", "String", optional=True),
                    FieldSpec("status", "String", default='"DRAFT"'),
                    FieldSpec("publishedAt", "DateTime", optional=True),
                    *AUDIT,
                ),
                (RelSpec("mn", "Tag", "tags"), RelSpec("1m", "ArticleVersion", "versions")),
            ),
            EntitySpec(
                "ArticleVersion", (FieldSpec("version", "Int"), FieldSpec("snapshot", "Json"), *AUDIT), ()
            ),
            EntitySpec("Tag", (FieldSpec("slug", "String", unique=True), FieldSpec("label", "String")), ()),
            EntitySpec(
                "Media",
                (
                    FieldSpec("url", "String"),
                    FieldSpec("alt", "String", optional=True),
                    FieldSpec("sizeBytes", "Int"),
                    *AUDIT,
                ),
                (),
            ),
        ),
        (EnumSpec("ArticleStatus", ("DRAFT", "IN_REVIEW", "PUBLISHED", "ARCHIVED")),),
    ),
    DomainSpec(
        "fintech_ledger",
        "Fintech ledger",
        (
            EntitySpec(
                "Tenant",
                (FieldSpec("slug", "String", unique=True), FieldSpec("name", "String"), *AUDIT),
                (RelSpec("1m", "Account", "accounts"), RelSpec("1m", "LedgerEntry", "entries")),
            ),
            EntitySpec(
                "Account",
                (
                    FieldSpec("number", "String", unique=True),
                    FieldSpec("kind", "String", default='"ASSET"'),
                    FieldSpec("currency", "String", default='"USD"'),
                    FieldSpec("balanceCents", "BigInt", default="0"),
                    *AUDIT,
                ),
                (RelSpec("1m", "LedgerEntry", "entries"),),
            ),
            EntitySpec(
                "LedgerEntry",
                (
                    FieldSpec("amountCents", "BigInt"),
                    FieldSpec("direction", "String"),
                    FieldSpec("memo", "String", optional=True),
                    FieldSpec("bookedAt", "DateTime", default="now()"),
                ),
                (),
            ),
            EntitySpec(
                "Transfer",
                (
                    FieldSpec("reference", "String", unique=True),
                    FieldSpec("status", "String", default='"PENDING"'),
                    FieldSpec("amountCents", "BigInt"),
                    *AUDIT,
                ),
                (RelSpec("1m", "AuditEvent", "auditEvents"),),
            ),
            EntitySpec(
                "AuditEvent",
                (
                    FieldSpec("action", "String"),
                    FieldSpec("payload", "Json", optional=True),
                    FieldSpec("at", "DateTime", default="now()"),
                ),
                (),
            ),
        ),
        (
            EnumSpec("EntryDirection", ("DEBIT", "CREDIT")),
            EnumSpec("TransferStatus", ("PENDING", "COMPLETED", "FAILED", "REVERSED")),
        ),
        tenant_field="tenantId",
    ),
    DomainSpec(
        "health",
        "Health clinic",
        (
            EntitySpec(
                "Patient",
                (
                    FieldSpec("mrn", "String", unique=True),
                    FieldSpec("fullName", "String"),
                    FieldSpec("dob", "DateTime"),
                    *AUDIT,
                ),
                (
                    RelSpec("1m", "Appointment", "appointments"),
                    RelSpec("1m", "Prescription", "prescriptions"),
                ),
            ),
            EntitySpec(
                "Provider",
                (
                    FieldSpec("npi", "String", unique=True),
                    FieldSpec("fullName", "String"),
                    FieldSpec("specialty", "String", optional=True),
                    *AUDIT,
                ),
                (RelSpec("1m", "Appointment", "appointments"),),
            ),
            EntitySpec(
                "Appointment",
                (
                    FieldSpec("scheduledAt", "DateTime"),
                    FieldSpec("status", "String", default='"BOOKED"'),
                    FieldSpec("reason", "String", optional=True),
                    *AUDIT,
                ),
                (),
            ),
            EntitySpec(
                "Prescription",
                (
                    FieldSpec("drug", "String"),
                    FieldSpec("dose", "String"),
                    FieldSpec("days", "Int", default="7"),
                    *AUDIT,
                ),
                (),
            ),
            EntitySpec(
                "Vital",
                (
                    FieldSpec("temperatureC", "Float", optional=True),
                    FieldSpec("systolic", "Int", optional=True),
                    FieldSpec("diastolic", "Int", optional=True),
                    FieldSpec("takenAt", "DateTime", default="now()"),
                ),
                (),
            ),
        ),
        (EnumSpec("AppointmentStatus", ("BOOKED", "CHECKED_IN", "COMPLETED", "CANCELLED")),),
    ),
    DomainSpec(
        "logistics",
        "Logistics network",
        (
            EntitySpec(
                "Warehouse",
                (
                    FieldSpec("code", "String", unique=True),
                    FieldSpec("city", "String"),
                    FieldSpec("capacity", "Int"),
                    *AUDIT,
                ),
                (RelSpec("1m", "StockLevel", "stockLevels"), RelSpec("1m", "Shipment", "outgoing")),
            ),
            EntitySpec(
                "StockLevel",
                (
                    FieldSpec("quantity", "Int", default="0"),
                    FieldSpec("reorderPoint", "Int", default="10"),
                    *AUDIT,
                ),
                (),
            ),
            EntitySpec(
                "Shipment",
                (
                    FieldSpec("trackingCode", "String", unique=True),
                    FieldSpec("status", "String", default='"CREATED"'),
                    FieldSpec("shippedAt", "DateTime", optional=True),
                    FieldSpec("deliveredAt", "DateTime", optional=True),
                    *AUDIT,
                ),
                (RelSpec("1m", "ShipmentEvent", "events"),),
            ),
            EntitySpec(
                "ShipmentEvent",
                (
                    FieldSpec("kind", "String"),
                    FieldSpec("at", "DateTime", default="now()"),
                    FieldSpec("note", "String", optional=True),
                ),
                (),
            ),
            EntitySpec(
                "Carrier",
                (
                    FieldSpec("name", "String", unique=True),
                    FieldSpec("contactPhone", "String", optional=True),
                ),
                (RelSpec("1m", "Shipment", "shipments"),),
            ),
        ),
        (EnumSpec("ShipmentStatus", ("CREATED", "IN_TRANSIT", "OUT_FOR_DELIVERY", "DELIVERED", "LOST")),),
    ),
    DomainSpec(
        "iot_telemetry",
        "IoT telemetry",
        (
            EntitySpec(
                "Device",
                (
                    FieldSpec("serial", "String", unique=True),
                    FieldSpec("model", "String"),
                    FieldSpec("firmware", "String", optional=True),
                    FieldSpec("lastSeenAt", "DateTime", optional=True),
                    *AUDIT,
                ),
                (RelSpec("1m", "Reading", "readings"), RelSpec("1m", "Alert", "alerts")),
            ),
            EntitySpec(
                "Reading",
                (
                    FieldSpec("metric", "String"),
                    FieldSpec("value", "Float"),
                    FieldSpec("recordedAt", "DateTime", default="now()"),
                ),
                (),
            ),
            EntitySpec(
                "Alert",
                (
                    FieldSpec("severity", "String", default='"INFO"'),
                    FieldSpec("message", "String"),
                    FieldSpec("acknowledgedAt", "DateTime", optional=True),
                    *AUDIT,
                ),
                (),
            ),
            EntitySpec(
                "Site",
                (FieldSpec("code", "String", unique=True), FieldSpec("region", "String"), *AUDIT),
                (RelSpec("1m", "Device", "devices"),),
            ),
            EntitySpec(
                "Firmware",
                (
                    FieldSpec("version", "String", unique=True),
                    FieldSpec("notes", "String", optional=True),
                    FieldSpec("releasedAt", "DateTime"),
                ),
                (),
            ),
        ),
        (EnumSpec("Severity", ("INFO", "WARN", "CRITICAL")),),
    ),
    DomainSpec(
        "lms",
        "Learning platform",
        (
            EntitySpec(
                "Student",
                (
                    FieldSpec("email", "String", unique=True),
                    FieldSpec("fullName", "String"),
                    FieldSpec("enrolledAt", "DateTime", default="now()"),
                    *AUDIT,
                ),
                (RelSpec("1m", "Enrollment", "enrollments"), RelSpec("1m", "Submission", "submissions")),
            ),
            EntitySpec(
                "Course",
                (
                    FieldSpec("code", "String", unique=True),
                    FieldSpec("title", "String"),
                    FieldSpec("description", "String", optional=True),
                    *AUDIT,
                ),
                (RelSpec("1m", "Lesson", "lessons"), RelSpec("1m", "Assignment", "assignments")),
            ),
            EntitySpec(
                "Lesson",
                (
                    FieldSpec("position", "Int"),
                    FieldSpec("title", "String"),
                    FieldSpec("content", "String", optional=True),
                    *AUDIT,
                ),
                (),
            ),
            EntitySpec(
                "Assignment",
                (
                    FieldSpec("title", "String"),
                    FieldSpec("dueAt", "DateTime", optional=True),
                    FieldSpec("maxPoints", "Int", default="100"),
                    *AUDIT,
                ),
                (RelSpec("1m", "Submission", "submissions"),),
            ),
            EntitySpec(
                "Enrollment",
                (
                    FieldSpec("status", "String", default='"ACTIVE"'),
                    FieldSpec("grade", "Float", optional=True),
                    *AUDIT,
                ),
                (),
            ),
            EntitySpec(
                "Submission",
                (
                    FieldSpec("content", "String", optional=True),
                    FieldSpec("score", "Int", optional=True),
                    FieldSpec("submittedAt", "DateTime", default="now()"),
                ),
                (),
            ),
        ),
        (EnumSpec("EnrollmentStatus", ("ACTIVE", "COMPLETED", "DROPPED", "WAITLISTED")),),
    ),
    DomainSpec(
        "analytics",
        "Product analytics",
        (
            EntitySpec(
                "Workspace",
                (FieldSpec("slug", "String", unique=True), FieldSpec("name", "String"), *AUDIT),
                (RelSpec("1m", "Project", "projects"), RelSpec("1m", "Dashboard", "dashboards")),
            ),
            EntitySpec(
                "Project",
                (FieldSpec("name", "String"), FieldSpec("apiKey", "String", unique=True), *AUDIT),
                (RelSpec("1m", "Event", "events"),),
            ),
            EntitySpec(
                "Event",
                (
                    FieldSpec("name", "String"),
                    FieldSpec("props", "Json", optional=True),
                    FieldSpec("occurredAt", "DateTime", default="now()"),
                ),
                (),
            ),
            EntitySpec(
                "Dashboard",
                (FieldSpec("title", "String"), FieldSpec("layout", "Json", optional=True), *AUDIT),
                (RelSpec("1m", "Widget", "widgets"),),
            ),
            EntitySpec(
                "Widget",
                (
                    FieldSpec("kind", "String"),
                    FieldSpec("query", "Json"),
                    FieldSpec("position", "Int", default="0"),
                ),
                (),
            ),
        ),
        (EnumSpec("WidgetKind", ("LINE", "BAR", "FUNNEL", "RETENTION", "NUMBER")),),
        tenant_field="organizationId",
    ),
    DomainSpec(
        "multi_tenant_hr",
        "Multi-tenant HR",
        (
            EntitySpec(
                "Company",
                (FieldSpec("slug", "String", unique=True), FieldSpec("legalName", "String"), *AUDIT),
                (
                    RelSpec("1m", "Department", "departments"),
                    RelSpec("1m", "Employee", "employees"),
                    RelSpec("1m", "PayrollRun", "payrollRuns"),
                ),
            ),
            EntitySpec(
                "Department",
                (FieldSpec("name", "String"), FieldSpec("costCenter", "String", optional=True), *AUDIT),
                (RelSpec("1m", "Employee", "employees"),),
            ),
            EntitySpec(
                "Employee",
                (
                    FieldSpec("employeeNumber", "String", unique=True),
                    FieldSpec("fullName", "String"),
                    FieldSpec("baseSalaryCents", "BigInt"),
                    FieldSpec("hiredAt", "DateTime"),
                    *AUDIT,
                ),
                (RelSpec("1m", "TimeOffRequest", "timeOffRequests"),),
            ),
            EntitySpec(
                "TimeOffRequest",
                (
                    FieldSpec("kind", "String", default='"VACATION"'),
                    FieldSpec("startsAt", "DateTime"),
                    FieldSpec("endsAt", "DateTime"),
                    FieldSpec("status", "String", default='"PENDING"'),
                    *AUDIT,
                ),
                (),
            ),
            EntitySpec(
                "PayrollRun",
                (
                    FieldSpec("periodStart", "DateTime"),
                    FieldSpec("periodEnd", "DateTime"),
                    FieldSpec("status", "String", default='"DRAFT"'),
                    *AUDIT,
                ),
                (RelSpec("1m", "Payslip", "payslips"),),
            ),
            EntitySpec(
                "Payslip",
                (
                    FieldSpec("grossCents", "BigInt"),
                    FieldSpec("netCents", "BigInt"),
                    FieldSpec("paidAt", "DateTime", optional=True),
                ),
                (),
            ),
        ),
        (
            EnumSpec("PayrollStatus", ("DRAFT", "APPROVED", "PAID", "CANCELLED")),
            EnumSpec("TimeOffKind", ("VACATION", "SICK", "UNPAID", "TRAINING")),
        ),
        tenant_field="companyId",
    ),
    DomainSpec(
        "marketplace",
        "Two-sided marketplace",
        (
            EntitySpec(
                "User",
                (
                    FieldSpec("email", "String", unique=True),
                    FieldSpec("displayName", "String"),
                    FieldSpec("isVendor", "Boolean", default="false"),
                    *AUDIT,
                ),
                (
                    RelSpec("1m", "Listing", "listings"),
                    RelSpec("1m", "Purchase", "purchases"),
                    RelSpec("1m", "Feedback", "feedbackGiven"),
                ),
            ),
            EntitySpec(
                "Listing",
                (
                    FieldSpec("title", "String"),
                    FieldSpec("description", "String", optional=True),
                    FieldSpec("priceCents", "Int"),
                    FieldSpec("stock", "Int", default="1"),
                    FieldSpec("status", "String", default='"ACTIVE"'),
                    *AUDIT,
                ),
                (RelSpec("1m", "Purchase", "purchases"),),
            ),
            EntitySpec(
                "Purchase",
                (
                    FieldSpec("quantity", "Int", default="1"),
                    FieldSpec("totalCents", "Int"),
                    FieldSpec("status", "String", default='"PENDING"'),
                    FieldSpec("purchasedAt", "DateTime", default="now()"),
                ),
                (RelSpec("1m", "Feedback", "feedback"),),
            ),
            EntitySpec(
                "Feedback",
                (FieldSpec("rating", "Int"), FieldSpec("comment", "String", optional=True), *AUDIT),
                (),
            ),
            EntitySpec(
                "Payout",
                (
                    FieldSpec("amountCents", "Int"),
                    FieldSpec("status", "String", default='"HELD"'),
                    FieldSpec("releasedAt", "DateTime", optional=True),
                    *AUDIT,
                ),
                (),
            ),
        ),
        (
            EnumSpec("ListingStatus", ("ACTIVE", "PAUSED", "SOLD_OUT")),
            EnumSpec("PurchaseStatus", ("PENDING", "PAID", "SHIPPED", "REFUNDED")),
        ),
    ),
)


def domain_by_key(key: str) -> DomainSpec:
    for d in DOMAINS_SPECS:
        if d.key == key:
            return d
    raise KeyError(f"unknown domain archetype '{key}'")
