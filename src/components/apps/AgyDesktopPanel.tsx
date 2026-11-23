import { useMemo, useState, type RefObject } from "react";
import { useTranslation } from "react-i18next";
import { ProviderList } from "@/components/providers/ProviderList";
import { ManagedAccountRemoveDialog } from "@/components/providers/forms/ManagedAccountRemoveDialog";
import { useManagedAccountUsers } from "@/components/providers/forms/hooks/useManagedAccountUsers";
import { useManagedAuth } from "@/components/providers/forms/hooks/useManagedAuth";
import { Button } from "@/components/ui/button";
import { Notice } from "@/components/ui/notice";
import type { Provider } from "@/types";

interface AgyDesktopPanelProps {
  savedProviders?: Record<string, Provider>;
  scrollRef?: RefObject<HTMLDivElement>;
  onCreate?: () => void;
  searchOpen?: boolean;
  onSearchOpenChange?: (open: boolean) => void;
}

export function AgyDesktopPanel({
  savedProviders,
  scrollRef,
  onCreate,
  searchOpen,
  onSearchOpenChange,
}: AgyDesktopPanelProps) {
  const { t } = useTranslation();
  const auth = useManagedAuth("agy_oauth", undefined, undefined, {
    agyDesktop: true,
  });
  const accountUsers = useManagedAccountUsers(
    "agy_oauth",
    auth.defaultAccountId,
  );
  const [removeTarget, setRemoveTarget] = useState<Provider | null>(null);
  const { providers, requiresReauth } = useMemo(() => {
    const providers: Record<string, Provider> = {};
    const requiresReauth = new Set<string>();
    for (const [index, account] of auth.accounts.entries()) {
      if (account.requires_reauth) requiresReauth.add(account.id);
      const savedProvider = savedProviders?.[account.id];
      if (savedProvider) {
        providers[account.id] = { ...savedProvider, sortIndex: index };
        continue;
      }
      providers[account.id] = {
        id: account.id,
        name: account.login,
        icon: "antigravity",
        category: "official",
        settingsConfig: {},
        sortIndex: index,
        meta: {
          authBinding: {
            source: "managed_account",
            authProvider: "agy_oauth",
            accountId: account.id,
          },
        },
      };
    }
    return { providers, requiresReauth };
  }, [auth.accounts, savedProviders]);

  return (
    <div
      ref={scrollRef}
      id="main-content"
      className="min-h-0 flex-1 overflow-y-auto scroll-stable overflow-x-hidden px-6 pb-12 pt-4"
    >
      {auth.isStatusError ? (
        <Notice
          tone="danger"
          title={t("authCenter.agy.statusLoadFailed")}
          actions={
            <Button
              variant="neutral"
              size="compact"
              onClick={() => void auth.refetchStatus()}
            >
              {t("authCenter.retry")}
            </Button>
          }
        />
      ) : (
        <ProviderList
          providers={providers}
          currentProviderId={auth.defaultAccountId ?? ""}
          appId="agy-desktop"
          isLoading={auth.isLoadingStatus}
          onSwitch={(provider) => auth.setDefaultAccount(provider.id)}
          onDelete={setRemoveTarget}
          onOpenWebsite={() => undefined}
          onCreate={onCreate}
          searchOpen={searchOpen}
          onSearchOpenChange={onSearchOpenChange}
          agyDesktopAccounts={{
            switching: auth.isSettingDefaultAccount,
            requiresReauth,
          }}
        />
      )}
      <ManagedAccountRemoveDialog
        target={
          removeTarget && {
            kind: "one",
            accountId: removeTarget.id,
            login: removeTarget.name,
          }
        }
        serviceName="Antigravity"
        users={removeTarget ? accountUsers([removeTarget.id]) : []}
        othersRemain={auth.accounts.length > 1}
        pending={auth.isRemovingAccount}
        onConfirm={() => {
          if (removeTarget) auth.removeAccount(removeTarget.id);
          setRemoveTarget(null);
        }}
        onCancel={() => setRemoveTarget(null)}
      />
    </div>
  );
}
